from rag.ingestion import load_pdfs
from rag.chunking import chunk_documents

documents = load_pdfs()
chunks = chunk_documents(documents)

print("\nFirst chunk:\n")
print(chunks[0]["text"])

print("\nMetadata:")
print(chunks[0]["metadata"])