import json
import math
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    BASE_DIR
    / "evaluation"
    / "processing"
    / "output"
    / "evaluation_chunks.json"
)

INDEX_FILE = (
    BASE_DIR
    / "evaluation"
    / "processing"
    / "output"
    / "evaluation_faiss.index"
)

DATASET_FILE = (
    BASE_DIR
    / "evaluation"
    / "dataset"
    / "evaluation_questions.json"
)

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

RESULTS_FILE = (
    RESULTS_DIR
    / "reranking_results.json"
)

TOP_K = 5

CANDIDATE_K = 10

RRF_K = 60

RERANK_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def precision_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    relevant_count = sum(
        1
        for chunk_id in retrieved_k
        if chunk_id in relevant
    )

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    relevant_count = sum(
        1
        for chunk_id in retrieved_k
        if chunk_id in relevant
    )

    if len(relevant) == 0:
        return 0.0

    return relevant_count / len(relevant)


def reciprocal_rank(retrieved, relevant):

    for rank, chunk_id in enumerate(
        retrieved,
        start=1
    ):

        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved_k,
        start=1
    ):

        if chunk_id in relevant:

            dcg += (
                1
                / math.log2(rank + 1)
            )

    ideal_relevant_count = min(
        len(relevant),
        k
    )

    if ideal_relevant_count == 0:
        return 0.0

    idcg = sum(
        1
        / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1
        )
    )

    return dcg / idcg


def main():

    print("=" * 60)
    print("HYBRID + CROSS-ENCODER RERANKING EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load chunks
    # ---------------------------------------------------------

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    print(
        "Evaluation chunks:",
        len(chunks)
    )

    # ---------------------------------------------------------
    # Load questions
    # ---------------------------------------------------------

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    print(
        "Evaluation questions:",
        len(questions)
    )

    # ---------------------------------------------------------
    # Load FAISS index
    # ---------------------------------------------------------

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print()
    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # ---------------------------------------------------------
    # Load Cross-Encoder
    # ---------------------------------------------------------

    print()
    print(
        "Loading Cross-Encoder..."
    )

    reranker_model = CrossEncoder(
        RERANK_MODEL
    )

    # ---------------------------------------------------------
    # Prepare BM25
    # ---------------------------------------------------------

    tokenized_documents = [
        tokenize(chunk["chunk_text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    # ---------------------------------------------------------
    # Metric storage
    # ---------------------------------------------------------

    all_results = []

    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    print()
    print(
        "Running evaluation..."
    )
    print()

    # =========================================================
    # Evaluate each question
    # =========================================================

    for question_data in questions:

        question_id = question_data[
            "question_id"
        ]

        question = question_data[
            "question"
        ]

        relevant_chunks = set(
            question_data[
                "relevant_chunks"
            ]
        )

        # =====================================================
        # 1. Dense Retrieval
        # =====================================================

        query_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        dense_distances, dense_indices = (
            index.search(
                query_embedding,
                CANDIDATE_K
            )
        )

        dense_results = [
            int(chunk_id)
            for chunk_id in dense_indices[0]
            if chunk_id != -1
        ]

        # =====================================================
        # 2. BM25 Retrieval
        # =====================================================

        tokenized_question = tokenize(
            question
        )

        bm25_scores = bm25.get_scores(
            tokenized_question
        )

        bm25_ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )

        bm25_results = [
            int(index)
            for index in bm25_ranked_indices[
                :CANDIDATE_K
            ]
        ]

        # =====================================================
        # 3. Reciprocal Rank Fusion
        # =====================================================

        rrf_scores = {}

        for rank, chunk_id in enumerate(
            dense_results,
            start=1
        ):

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0

            rrf_scores[chunk_id] += (
                1
                / (RRF_K + rank)
            )

        for rank, chunk_id in enumerate(
            bm25_results,
            start=1
        ):

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0

            rrf_scores[chunk_id] += (
                1
                / (RRF_K + rank)
            )

        hybrid_ranked = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        hybrid_candidates = [
            chunk_id
            for chunk_id, score
            in hybrid_ranked[
                :CANDIDATE_K
            ]
        ]

        # =====================================================
        # 4. Cross-Encoder Reranking
        # =====================================================

        pairs = []

        for chunk_id in hybrid_candidates:

            chunk_text = chunks[
                chunk_id
            ]["chunk_text"]

            pairs.append(
                (
                    question,
                    chunk_text
                )
            )

        reranker_scores = (
            reranker_model.predict(
                pairs
            )
        )

        reranked = []

        for chunk_id, score in zip(
            hybrid_candidates,
            reranker_scores
        ):

            reranked.append(
                {
                    "chunk_id": chunk_id,
                    "score": float(score)
                }
            )

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Final top 5
        retrieved_chunk_ids = [
            item["chunk_id"]
            for item in reranked[
                :TOP_K
            ]
        ]

        # =====================================================
        # 5. Calculate Metrics
        # =====================================================

        precision = precision_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K
        )

        recall = recall_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K
        )

        mrr = reciprocal_rank(
            retrieved_chunk_ids,
            relevant_chunks
        )

        ndcg = ndcg_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        mrr_scores.append(
            mrr
        )

        ndcg_scores.append(
            ndcg
        )

        # =====================================================
        # Save question result
        # =====================================================

        result = {
            "question_id": question_id,
            "question": question,
            "relevant_chunks": sorted(
                relevant_chunks
            ),
            "hybrid_candidates": hybrid_candidates,
            "reranked_chunks": retrieved_chunk_ids,
            "reranker_scores": [
                item
                for item in reranked[
                    :TOP_K
                ]
            ],
            "precision_at_5": precision,
            "recall_at_5": recall,
            "mrr": mrr,
            "ndcg_at_5": ndcg
        }

        all_results.append(
            result
        )

        print(
            f"{question_id} | "
            f"P@5={precision:.3f} | "
            f"R@5={recall:.3f} | "
            f"MRR={mrr:.3f} | "
            f"NDCG@5={ndcg:.3f}"
        )

    # =========================================================
    # Average metrics
    # =========================================================

    question_count = len(
        questions
    )

    average_precision = (
        sum(precision_scores)
        / question_count
    )

    average_recall = (
        sum(recall_scores)
        / question_count
    )

    average_mrr = (
        sum(mrr_scores)
        / question_count
    )

    average_ndcg = (
        sum(ndcg_scores)
        / question_count
    )

    # =========================================================
    # Summary
    # =========================================================

    summary = {
        "method": (
            "Hybrid Retrieval + "
            "Cross-Encoder Reranking"
        ),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_constant": RRF_K,
        "reranker_model": RERANK_MODEL,
        "number_of_questions": question_count,
        "precision_at_5": average_precision,
        "recall_at_5": average_recall,
        "mrr": average_mrr,
        "ndcg_at_5": average_ndcg
    }

    # =========================================================
    # Save results
    # =========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "summary": summary,
        "questions": all_results
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # =========================================================
    # Print final results
    # =========================================================

    print()
    print("=" * 60)
    print(
        "HYBRID + CROSS-ENCODER RERANKING RESULTS"
    )
    print("=" * 60)

    print(
        f"Questions:     {question_count}"
    )

    print(
        f"Precision@5:   {average_precision:.4f}"
    )

    print(
        f"Recall@5:      {average_recall:.4f}"
    )

    print(
        f"MRR:           {average_mrr:.4f}"
    )

    print(
        f"NDCG@5:        {average_ndcg:.4f}"
    )

    print()
    print(
        "Results saved to:"
    )
    print(
        RESULTS_FILE
    )


if __name__ == "__main__":
    main()