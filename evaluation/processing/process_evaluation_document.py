import json
import re
from pathlib import Path

import fitz


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

PDF_FILE = BASE_DIR / "evaluation" / "documents" / "cloud_serverless.pdf"

OUTPUT_DIR = BASE_DIR / "evaluation" / "processing" / "output"

CHUNKS_FILE = OUTPUT_DIR / "evaluation_chunks.json"


def clean_text(text):
    """
    Clean extracted PDF text.
    """

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def create_chunks(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_document():

    if not PDF_FILE.exists():
        print("ERROR: Evaluation PDF not found.")
        print(PDF_FILE)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document = fitz.open(PDF_FILE)

    all_chunks = []

    chunk_counter = 0

    print("Processing evaluation document...")
    print("PDF:", PDF_FILE)
    print("Pages:", len(document))

    for page_number, page in enumerate(document, start=1):

        raw_text = page.get_text()

        cleaned_text = clean_text(raw_text)

        page_chunks = create_chunks(
            cleaned_text,
            chunk_size=1000,
            overlap=200
        )

        for chunk in page_chunks:

            all_chunks.append({
                "document_name": "cloud_serverless.pdf",
                "page_number": page_number,
                "chunk_id": chunk_counter,
                "chunk_text": chunk
            })

            chunk_counter += 1

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Processing completed successfully.")
    print("Total pages:", len(document))
    print("Total chunks:", len(all_chunks))
    print("Chunks saved to:")
    print(CHUNKS_FILE)


if __name__ == "__main__":
    process_document()