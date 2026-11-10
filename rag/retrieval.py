
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag.embeddings import get_embedding_model


# ==========================================
# RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve_documents(query, chunks, embeddings, top_k=3):

    if not chunks or len(embeddings) == 0:
        return []

    # Load the embedding model when needed
    model = get_embedding_model()

    # Convert query into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Get top matching document indexes
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index]["text"],
            "metadata": chunks[index]["metadata"],
            "score": float(similarity_scores[index])
        })

    return results
