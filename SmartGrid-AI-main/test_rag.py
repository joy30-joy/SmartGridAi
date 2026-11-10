from rag.ingestion import load_pdfs
from rag.chunking import chunk_documents
from rag.embeddings import generate_embeddings
from rag.retrieval import retrieve_documents
from rag.llm import generate_answer


# Step 1: Load PDFs
documents = load_pdfs()

# Step 2: Create chunks
chunks = chunk_documents(documents)

# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 4: Ask a question
query = "Why can extreme heat increase electricity demand in Texas?"

# Step 5: Retrieve relevant information
results = retrieve_documents(
    query,
    chunks,
    embeddings,
    top_k=3
)

# Step 6: Generate answer using Gemini
answer = generate_answer(query, results)

print("\nSMARTGRID AI ANSWER:\n")
print(answer)