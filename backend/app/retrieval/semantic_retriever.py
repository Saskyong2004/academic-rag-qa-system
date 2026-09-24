import json
from pathlib import Path

import faiss
import numpy as np

from app.services.embedding_service import generate_embeddings


CHUNKS_FILE = Path("app/data/chunks.json")
INDEX_FILE = Path("app/data/faiss.index")


def build_faiss_index():
    """Create a FAISS index from stored document chunks."""

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    texts = [chunk["chunk_text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, str(INDEX_FILE))

    return len(chunks), dimension

def semantic_search(question, top_k=3):
    """Search the FAISS index for semantically similar chunks."""

    if not INDEX_FILE.exists():
        return []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    index = faiss.read_index(str(INDEX_FILE))

    question_embedding = generate_embeddings([question])

    question_vector = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_vector, top_k)

    results = []

    for distance, index_number in zip(distances[0], indices[0]):
        if index_number == -1:
            continue

        chunk = chunks[index_number]

        results.append({
            "score": float(distance),
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
        })

    return results