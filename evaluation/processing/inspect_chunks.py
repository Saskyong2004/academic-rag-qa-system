import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    BASE_DIR
    / "evaluation"
    / "processing"
    / "output"
    / "evaluation_chunks.json"
)


def inspect_chunks():

    if not CHUNKS_FILE.exists():
        print("ERROR: evaluation_chunks.json not found.")
        return

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print("=" * 80)
    print("EVALUATION DOCUMENT CHUNKS")
    print("=" * 80)

    print(f"Total chunks: {len(chunks)}")
    print()

    for chunk in chunks:

        print("-" * 80)

        print(
            f"Chunk ID: {chunk['chunk_id']} | "
            f"Page: {chunk['page_number']}"
        )

        print()

        print(chunk["chunk_text"])

        print()


if __name__ == "__main__":
    inspect_chunks()