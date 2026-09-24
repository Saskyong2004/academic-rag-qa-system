import json
import re
from pathlib import Path


CHUNKS_FILE = Path("app/data/chunks.json")


def load_chunks():
    """Load processed chunks from storage."""

    if not CHUNKS_FILE.exists():
        return []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text):
    """Convert text into simple lowercase words."""

    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def retrieve_chunks(question, top_k=3):
    """Retrieve the most relevant chunks using keyword overlap."""

    chunks = load_chunks()

    if not chunks:
        return []

    question_words = set(tokenize(question))

    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(tokenize(chunk["chunk_text"]))

        score = len(question_words.intersection(chunk_words))

        if score > 0:
            scored_chunks.append({
                "score": score,
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
            })

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_chunks[:top_k]