import json
import math
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    BASE_DIR
    / "evaluation"
    / "processing"
    / "output"
    / "evaluation_chunks.json"
)

DATASET_FILE = (
    BASE_DIR
    / "evaluation"
    / "dataset"
    / "evaluation_questions.json"
)

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

RESULTS_FILE = RESULTS_DIR / "bm25_results.json"

TOP_K = 5


def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved_k
        if chunk_id in relevant
    )

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved_k
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
            dcg += 1 / math.log2(rank + 1)

    ideal_relevant_count = min(
        len(relevant),
        k
    )

    if ideal_relevant_count == 0:
        return 0.0

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1
        )
    )

    return dcg / idcg


def main():

    print("=" * 60)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load evaluation chunks
    # ---------------------------------------------------------

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print("Evaluation chunks:", len(chunks))

    # ---------------------------------------------------------
    # Load evaluation questions
    # ---------------------------------------------------------

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    print("Evaluation questions:", len(questions))

    # ---------------------------------------------------------
    # Prepare BM25 corpus
    # ---------------------------------------------------------

    tokenized_documents = [
        tokenize(chunk["chunk_text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_documents)

    # ---------------------------------------------------------
    # Metric storage
    # ---------------------------------------------------------

    all_results = []

    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    print()
    print("Running evaluation...")
    print()

    # ---------------------------------------------------------
    # Evaluate every question
    # ---------------------------------------------------------

    for question_data in questions:

        question_id = question_data["question_id"]

        question = question_data["question"]

        relevant_chunks = set(
            question_data["relevant_chunks"]
        )

        # Tokenize question
        tokenized_question = tokenize(question)

        # Calculate BM25 scores
        scores = bm25.get_scores(
            tokenized_question
        )

        # Sort chunk indices by BM25 score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        # Top K chunks
        retrieved_chunk_ids = [
            int(index)
            for index in ranked_indices[:TOP_K]
        ]

        # -----------------------------------------------------
        # Calculate metrics
        # -----------------------------------------------------

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

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)

        # -----------------------------------------------------
        # Save individual result
        # -----------------------------------------------------

        result = {
            "question_id": question_id,
            "question": question,
            "relevant_chunks": sorted(
                relevant_chunks
            ),
            "retrieved_chunks": retrieved_chunk_ids,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "mrr": mrr,
            "ndcg_at_5": ndcg
        }

        all_results.append(result)

        print(
            f"{question_id} | "
            f"P@5={precision:.3f} | "
            f"R@5={recall:.3f} | "
            f"MRR={mrr:.3f} | "
            f"NDCG@5={ndcg:.3f}"
        )

    # ---------------------------------------------------------
    # Calculate average metrics
    # ---------------------------------------------------------

    question_count = len(questions)

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

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "method": "BM25",
        "top_k": TOP_K,
        "number_of_questions": question_count,
        "precision_at_5": average_precision,
        "recall_at_5": average_recall,
        "mrr": average_mrr,
        "ndcg_at_5": average_ndcg
    }

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("BM25 RETRIEVAL RESULTS")
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
    print("Results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()