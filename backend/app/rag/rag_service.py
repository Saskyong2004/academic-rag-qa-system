from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank_results
from app.services.llm_service import generate_answer


def answer_question(question, top_k=5):
    """
    Final RAG pipeline:

    Question
        ↓
    Hybrid Retrieval
        ↓
    Cross-Encoder Reranking
        ↓
    Relevant Context
        ↓
    Gemini
        ↓
    Grounded Answer + Sources
    """

    # Step 1: Retrieve candidates using Hybrid Retrieval
    hybrid_results = hybrid_search(question, top_k=5)

    if not hybrid_results:
        return {
            "status": "error",
            "message": "No relevant document context found."
        }

    # Step 2: Rerank Hybrid candidates
    retrieved_chunks = rerank_results(
        question,
        hybrid_results,
        top_k=top_k
    )

    if not retrieved_chunks:
        return {
            "status": "error",
            "message": "No relevant document context found after reranking."
        }

    # Step 3: Build context for Gemini
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

    # Step 4: Generate grounded answer
    answer = generate_answer(
        question=question,
        context=context
    )

    # Step 5: Prepare source information
    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["reranker_score"],
            "chunk_text": chunk["chunk_text"]
        })

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "sources": sources
    }