import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = BASE_DIR / "evaluation" / "processing" / "output" / "evaluation_chunks.json"
INDEX_FILE = BASE_DIR / "evaluation" / "processing" / "output" / "evaluation_faiss.index"
QUESTIONS_FILE = BASE_DIR / "evaluation" / "dataset" / "evaluation_questions.json"
RESULTS_FILE = BASE_DIR / "evaluation" / "results" / "rag_answers.json"

ENV_FILE = BASE_DIR / "backend" / ".env"


# ============================================================
# CONFIGURATION
# ============================================================

DENSE_CANDIDATES = 10
BM25_CANDIDATES = 10
RERANK_CANDIDATES = 10
FINAL_TOP_K = 5

RRF_K = 60

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading evaluation data...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

index = faiss.read_index(str(INDEX_FILE))

print(f"Loaded {len(chunks)} chunks.")
print(f"Loaded {len(questions)} evaluation questions.")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading cross-encoder reranker...")

reranker_model = CrossEncoder(RERANKER_MODEL)

print("Models loaded.")


# ============================================================
# BM25 SETUP
# ============================================================

def tokenize(text):
    import re
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


tokenized_documents = [
    tokenize(chunk["chunk_text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_documents)


# ============================================================
# GEMINI SETUP
# ============================================================

load_dotenv(ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in backend/.env"
    )

client = genai.Client(api_key=api_key)


# ============================================================
# CHECKPOINT / RESUME FUNCTIONS
# ============================================================

def load_existing_results():
    """
    Load previously completed questions.

    Returns a dictionary:
        {
            "Q01": {...},
            "Q02": {...}
        }
    """

    if not RESULTS_FILE.exists():
        return {}

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Convert old list format into dictionary if necessary
        if isinstance(data, list):
            return {
                item["question_id"]: item
                for item in data
                if "question_id" in item
            }

        if isinstance(data, dict):
            return data

    except (json.JSONDecodeError, OSError) as error:
        print(f"Warning: Could not read existing results: {error}")

    return {}


def save_results(results):
    """
    Save ALL completed results immediately.
    """

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save in question order
    ordered_results = []

    for question in questions:
        question_id = question["question_id"]

        if question_id in results:
            ordered_results.append(results[question_id])

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            ordered_results,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# RETRIEVAL
# ============================================================

def dense_search(question, top_k=DENSE_CANDIDATES):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
            "dense_distance": float(distance)
        })

    return results


def bm25_search(question, top_k=BM25_CANDIDATES):

    scores = bm25.get_scores(tokenize(question))

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in ranked_indices:

        chunk = chunks[int(idx)]

        results.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
            "bm25_score": float(scores[idx])
        })

    return results


def hybrid_search(question):

    dense_results = dense_search(
        question,
        DENSE_CANDIDATES
    )

    bm25_results = bm25_search(
        question,
        BM25_CANDIDATES
    )

    combined = {}

    # -------------------------
    # Dense rankings
    # -------------------------

    for rank, result in enumerate(dense_results, start=1):

        key = (
            result["document_name"],
            result["page_number"],
            result["chunk_id"]
        )

        combined[key] = {
            "document_name": result["document_name"],
            "page_number": result["page_number"],
            "chunk_id": result["chunk_id"],
            "chunk_text": result["chunk_text"],
            "dense_rank": rank,
            "bm25_rank": None,
            "hybrid_score": 1 / (RRF_K + rank)
        }

    # -------------------------
    # BM25 rankings
    # -------------------------

    for rank, result in enumerate(bm25_results, start=1):

        key = (
            result["document_name"],
            result["page_number"],
            result["chunk_id"]
        )

        if key not in combined:

            combined[key] = {
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "dense_rank": None,
                "bm25_rank": rank,
                "hybrid_score": 0.0
            }

        else:

            combined[key]["bm25_rank"] = rank

        combined[key]["hybrid_score"] += (
            1 / (RRF_K + rank)
        )

    results = list(combined.values())

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return results


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(question, results):

    if not results:
        return []

    candidates = results[:RERANK_CANDIDATES]

    pairs = [
        (question, result["chunk_text"])
        for result in candidates
    ]

    scores = reranker_model.predict(pairs)

    reranked = []

    for result, score in zip(candidates, scores):

        reranked.append({
            "document_name": result["document_name"],
            "page_number": result["page_number"],
            "chunk_id": result["chunk_id"],
            "chunk_text": result["chunk_text"],
            "reranker_score": float(score)
        })

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked[:FINAL_TOP_K]


# ============================================================
# GEMINI ANSWER GENERATION
# ============================================================

def generate_answer(question, retrieved_chunks):

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Document: {chunk["document_name"]}
Page: {chunk["page_number"]}
Chunk ID: {chunk["chunk_id"]}

{chunk["chunk_text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are an academic question-answering assistant.

Answer the user's question using ONLY the provided document context.

Do not use outside knowledge.

If the answer cannot be found in the provided context, clearly say:

"I could not find the answer in the provided document."

Do not invent or assume information.

For the answer, provide a concise and clear explanation.

Where appropriate, mention the relevant page number from the provided context.

User Question:
{question}

Document Context:
{context}
"""

    response = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        generation_config={
            "thinking_level": "low"
        }
    )

    return response.output_text


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2 — END-TO-END RAG EVALUATION")
print("=" * 70)

results = load_existing_results()

completed_ids = set(results.keys())

print(f"\nPreviously completed questions: {len(completed_ids)}")

if completed_ids:

    print(
        "Already completed:",
        ", ".join(
            q["question_id"]
            for q in questions
            if q["question_id"] in completed_ids
        )
    )

remaining_questions = [
    q
    for q in questions
    if q["question_id"] not in completed_ids
]

print(f"Remaining questions: {len(remaining_questions)}")


if not remaining_questions:

    print("\nAll 20 questions are already completed.")
    print(f"Results file: {RESULTS_FILE}")
    print("Nothing more to run.")

else:

    for question_number, item in enumerate(
        remaining_questions,
        start=1
    ):

        question_id = item["question_id"]
        question = item["question"]

        print("\n" + "-" * 70)
        print(
            f"Processing {question_id}: "
            f"{question}"
        )
        print("-" * 70)

        start_time = time.perf_counter()

        try:

            # ------------------------------------------
            # STEP 1 — Hybrid retrieval
            # ------------------------------------------

            hybrid_results = hybrid_search(question)

            print(
                f"Hybrid candidates: "
                f"{len(hybrid_results)}"
            )

            # ------------------------------------------
            # STEP 2 — Cross-encoder reranking
            # ------------------------------------------

            reranked_results = rerank_results(
                question,
                hybrid_results
            )

            print(
                f"Reranked chunks: "
                f"{len(reranked_results)}"
            )

            if not reranked_results:

                print(
                    f"ERROR: No retrieved context for "
                    f"{question_id}"
                )

                continue

            # ------------------------------------------
            # STEP 3 — Gemini
            # ------------------------------------------

            print("Generating Gemini answer...")

            answer = generate_answer(
                question,
                reranked_results
            )

            # ------------------------------------------
            # STEP 4 — Latency
            # ------------------------------------------

            latency = time.perf_counter() - start_time

            # ------------------------------------------
            # STEP 5 — Store result
            # ------------------------------------------

            sources = []

            for chunk in reranked_results:

                sources.append({
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["reranker_score"],
                    "chunk_text": chunk["chunk_text"]
                })

            result = {
                "question_id": question_id,
                "question": question,

                "relevant_chunks": item[
                    "relevant_chunks"
                ],

                "retrieval_method": (
                    "Hybrid RRF + Cross-Encoder Reranking"
                ),

                "dense_candidates": DENSE_CANDIDATES,
                "bm25_candidates": BM25_CANDIDATES,
                "rerank_candidates": RERANK_CANDIDATES,
                "final_top_k": FINAL_TOP_K,

                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "llm_model": GEMINI_MODEL,

                "answer": answer,

                "sources": sources,

                "latency_seconds": round(
                    latency,
                    4
                )
            }

            # ------------------------------------------
            # CRITICAL:
            # SAVE IMMEDIATELY
            # ------------------------------------------

            results[question_id] = result

            save_results(results)

            print(
                f"\nSUCCESS: {question_id} saved."
            )

            print(
                f"Latency: {latency:.2f} seconds"
            )

            print(
                f"Progress: "
                f"{len(results)}/{len(questions)}"
            )

        except Exception as error:

            error_text = str(error)

            print(
                f"\nERROR while processing "
                f"{question_id}:"
            )

            print(error_text)

            # ------------------------------------------
            # RATE LIMIT / QUOTA ERROR
            # ------------------------------------------

            if (
                "429" in error_text
                or "RateLimitError" in error_text
                or "rate limit" in error_text.lower()
                or "quota" in error_text.lower()
            ):

                print("\nGemini quota/rate limit detected.")

                # Save whatever has already completed
                save_results(results)

                print(
                    f"Checkpoint saved successfully."
                )

                print(
                    f"Completed questions: "
                    f"{len(results)}/{len(questions)}"
                )

                print(
                    "\nStop now and wait for the quota "
                    "to reset."
                )

                print(
                    "Run this same script again later "
                    "to resume."
                )

                break

            # ------------------------------------------
            # OTHER ERROR
            # ------------------------------------------

            print(
                "\nUnexpected error encountered."
            )

            print(
                "Saving current checkpoint..."
            )

            save_results(results)

            print(
                "Checkpoint saved."
            )

            print(
                "You can run the script again "
                "after fixing the error."
            )

            break


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("CHECKPOINT STATUS")
print("=" * 70)

print(
    f"Completed: {len(results)}/{len(questions)}"
)

missing = [
    q["question_id"]
    for q in questions
    if q["question_id"] not in results
]

if missing:

    print(
        "Remaining:",
        ", ".join(missing)
    )

    print(
        "\nRun the script again later to continue."
    )

else:

    print(
        "\nALL 20 QUESTIONS COMPLETED!"
    )

    print(
        f"Results saved to:\n{RESULTS_FILE}"
    )

print("=" * 70)