from rag.ingestion import load_pdfs
from rag.chunking import chunk_documents
from rag.embeddings import generate_embeddings
from rag.retrieval import retrieve_documents

documents = load_pdfs()
chunks = chunk_documents(documents)

embeddings = generate_embeddings(chunks)

query = "What weather conditions can increase electricity demand in Texas?"

results = retrieve_documents(query, chunks, embeddings)

print("\nUSER QUERY:")
print(query)

print("\nRETRIEVED DOCUMENTS:")

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", result["metadata"]["source"])
    print("Page:", result["metadata"]["page"])
    print("Similarity:", round(result["score"], 4))
    print("Text:", result["text"][:500])