from app.retrieval.semantic_retriever import semantic_search
from app.retrieval.bm25_retriever import bm25_search


def hybrid_search(question, top_k=5):
    """
    Hybrid retrieval using:
    1. Dense semantic retrieval (FAISS)
    2. BM25 keyword retrieval
    3. Reciprocal Rank Fusion (RRF)
    """

    # Get candidates from both retrieval methods
    dense_results = semantic_search(question, top_k=5)
    bm25_results = bm25_search(question, top_k=5)

    combined_results = {}

    # Add Dense retrieval results
    for rank, result in enumerate(dense_results, start=1):

        key = (
            result["document_name"],
            result["page_number"],
            result["chunk_id"]
        )

        if key not in combined_results:
            combined_results[key] = {
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "dense_rank": None,
                "bm25_rank": None,
                "hybrid_score": 0.0
            }

        combined_results[key]["dense_rank"] = rank

        # RRF contribution from Dense retrieval
        combined_results[key]["hybrid_score"] += 1 / (60 + rank)

    # Add BM25 retrieval results
    for rank, result in enumerate(bm25_results, start=1):

        key = (
            result["document_name"],
            result["page_number"],
            result["chunk_id"]
        )

        if key not in combined_results:
            combined_results[key] = {
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "dense_rank": None,
                "bm25_rank": None,
                "hybrid_score": 0.0
            }

        combined_results[key]["bm25_rank"] = rank

        # RRF contribution from BM25 retrieval
        combined_results[key]["hybrid_score"] += 1 / (60 + rank)

    # Convert dictionary to list
    results = list(combined_results.values())

    # Highest RRF score = highest combined rank
    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return results[:top_k]