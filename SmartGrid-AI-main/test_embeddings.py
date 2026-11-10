from rag.ingestion import load_pdfs
from rag.chunking import chunk_documents
from rag.embeddings import generate_embeddings

documents = load_pdfs()
chunks = chunk_documents(documents)

embeddings = generate_embeddings(chunks)

print("\nFirst embedding:")
print(embeddings[0][:10])