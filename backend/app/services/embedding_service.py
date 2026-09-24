from sentence_transformers import SentenceTransformer


# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Generate semantic embeddings for a list of texts.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings