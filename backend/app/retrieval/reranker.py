from sentence_transformers import CrossEncoder


# Load the reranking model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(question, results, top_k=5):
    """
    Rerank retrieved document chunks using a Cross-Encoder.

    The Cross-Encoder evaluates the question and each
    retrieved passage together and assigns a relevance score.
    """

    if not results:
        return []

    # Create question-passage pairs
    pairs = [
        (question, result["chunk_text"])
        for result in results
    ]

    # Generate relevance scores
    scores = reranker_model.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):
        reranked_results.append({
            "document_name": result["document_name"],
            "page_number": result["page_number"],
            "chunk_id": result["chunk_id"],
            "chunk_text": result["chunk_text"],
            "reranker_score": float(score)
        })

    # Higher reranker score = more relevant
    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]