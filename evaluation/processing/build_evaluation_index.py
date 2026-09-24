import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


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


def build_index():

    if not CHUNKS_FILE.exists():
        print("ERROR: evaluation_chunks.json not found.")
        return

    print("=" * 60)
    print("BUILDING EVALUATION FAISS INDEX")
    print("=" * 60)

    # Load evaluation chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Total chunks:", len(chunks))

    texts = [chunk["chunk_text"] for chunk in chunks]

    # Load embedding model
    print()
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("Embedding shape:", embeddings.shape)

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    # Save index
    faiss.write_index(index, str(INDEX_FILE))

    print()
    print("FAISS index built successfully.")
    print("Index dimension:", dimension)
    print("Vectors stored:", index.ntotal)
    print("Saved to:")
    print(INDEX_FILE)


if __name__ == "__main__":
    build_index()