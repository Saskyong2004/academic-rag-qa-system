import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("app/data/chunks.json")


def load_chunks():
    if not CHUNKS_FILE.exists():
        return []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def bm25_search(question, top_k=5):
    chunks = load_chunks()

    if not chunks:
        return []

    # Prepare documents for BM25
    tokenized_documents = [
        tokenize(chunk["chunk_text"])
        for chunk in chunks
    ]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_documents)

    # Tokenize the user's question
    tokenized_question = tokenize(question)

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_question)

    # Combine scores with chunk information
    scored_chunks = []

    for index, score in enumerate(scores):
        scored_chunks.append({
            "score": float(score),
            "document_name": chunks[index]["document_name"],
            "page_number": chunks[index]["page_number"],
            "chunk_id": chunks[index]["chunk_id"],
            "chunk_text": chunks[index]["chunk_text"],
        })

    # Highest BM25 score = most relevant
    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_chunks[:top_k]