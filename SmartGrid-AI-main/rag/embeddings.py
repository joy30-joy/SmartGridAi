
from sentence_transformers import SentenceTransformer
import numpy as np

# ==========================================
# EMBEDDING MODEL
# ==========================================

_model = None

def get_embedding_model():

    global _model

    if _model is None:

        print("\nLoading Sentence Transformer model...")

        _model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully!")

    return _model


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):

    if not chunks:
        raise ValueError("No chunks provided for embedding!")

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("\nEmbeddings generated successfully!")
    print("Embedding shape:", embeddings.shape)

    return embeddings
