def chunk_documents(documents, chunk_size=500, overlap=100):
    chunks = []

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata.copy()
                })

            start += chunk_size - overlap

    print(f"Total chunks created: {len(chunks)}")

    return chunks