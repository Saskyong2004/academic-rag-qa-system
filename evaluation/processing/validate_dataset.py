import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    BASE_DIR
    / "evaluation"
    / "dataset"
    / "evaluation_questions.json"
)


def validate_dataset():

    if not DATASET_FILE.exists():
        print("ERROR: evaluation_questions.json not found.")
        return

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    print("=" * 60)
    print("EVALUATION DATASET VALIDATION")
    print("=" * 60)

    print("Total questions:", len(questions))

    question_ids = [
        question["question_id"]
        for question in questions
    ]

    duplicate_ids = {
        question_id
        for question_id in question_ids
        if question_ids.count(question_id) > 1
    }

    if duplicate_ids:
        print("ERROR: Duplicate question IDs:", duplicate_ids)
        return

    invalid_chunks = []

    for question in questions:

        for chunk_id in question["relevant_chunks"]:

            if not isinstance(chunk_id, int) or chunk_id < 0 or chunk_id > 24:
                invalid_chunks.append(
                    (
                        question["question_id"],
                        chunk_id
                    )
                )

    if invalid_chunks:
        print("ERROR: Invalid chunk references:")
        print(invalid_chunks)
        return

    print("Question IDs: unique")
    print("Chunk references: valid")
    print("Dataset validation: SUCCESS")


if __name__ == "__main__":
    validate_dataset()