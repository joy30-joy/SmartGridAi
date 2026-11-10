
from ml.prediction import predict_grid_status

from rag.ingestion import load_pdfs
from rag.chunking import chunk_documents
from rag.embeddings import generate_embeddings
from rag.retrieval import retrieve_documents
from rag.llm import generate_answer


# ==========================================
# 1. KNOWLEDGE BASE CACHE
# ==========================================

_chunks = None
_embeddings = None


def load_knowledge_base():

    global _chunks, _embeddings

    # Reuse knowledge base if already loaded
    if _chunks is not None and _embeddings is not None:
        print("\nUsing cached knowledge base...")
        return _chunks, _embeddings

    print("\nLoading PDF knowledge base...")

    # STEP 1: Extract PDF text
    documents = load_pdfs()

    if not documents:
        raise ValueError(
            "No PDF documents found in the documents folder!"
        )

    print("PDF documents loaded:", len(documents))

    # STEP 2: Create chunks
    print("\nCreating text chunks...")

    _chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100
    )

    if not _chunks:
        raise ValueError("No text chunks were created!")

    print("Total chunks created:", len(_chunks))

    # STEP 3: Generate embeddings
    print("\nGenerating embeddings...")

    _embeddings = generate_embeddings(_chunks)

    print("Knowledge base loaded successfully!")

    return _chunks, _embeddings


# ==========================================
# 2. COMPLETE SMARTGRID PIPELINE
# ==========================================

def run_smartgrid(input_data):

    print("\n========== SMARTGRID AI ==========")

    # STEP 1: ML PREDICTIONS
    print("\nStep 1: Running ML models...")

    prediction = predict_grid_status(input_data)

    print("Predicted usage:", prediction["predicted_usage"])
    print("Risk status:", prediction["risk_status"])
    print(
        "Risk probability:",
        prediction["risk_probability"],
        "%"
    )
    print("Usage threshold:", prediction["usage_threshold"])

    # STEP 2: LOAD KNOWLEDGE BASE
    print("\nStep 2: Loading knowledge base...")

    chunks, embeddings = load_knowledge_base()

    # STEP 3: RETRIEVE RELEVANT DOCUMENTS
    print("\nStep 3: Retrieving relevant documents...")

    retrieved_results = retrieve_documents(
        prediction["rag_query"],
        chunks,
        embeddings,
        top_k=3
    )

    print("Documents retrieved:", len(retrieved_results))

    if not retrieved_results:
        raise ValueError(
            "No relevant documents were retrieved!"
        )

    # STEP 4: GENERATE GEMINI EXPLANATION
    print("\nStep 4: Generating explanation using Gemini...")

    answer = generate_answer(
        prediction["rag_query"],
        retrieved_results
    )

    # STEP 5: PREPARE FINAL RESULT
    result = {
        "predicted_usage": prediction["predicted_usage"],
        "risk_status": prediction["risk_status"],
        "risk_probability": prediction["risk_probability"],
        "usage_threshold": prediction["usage_threshold"],
        "explanation": answer,

        "sources": [
            {
                "document": item["metadata"]["source"],
                "page": item["metadata"]["page"],
                "similarity_score": round(
                    float(item["score"]), 4
                )
            }
            for item in retrieved_results
        ]
    }

    print("\nSmartGrid AI pipeline completed!")

    return result


# ==========================================
# 3. TEST COMPLETE PIPELINE
# ==========================================

if __name__ == "__main__":

    sample_input = {
        "latitude": 32.7767,
        "longitude": -96.7970,
        "ERCOT_zone": "NORTH",
        "date": "2026-07-15",
        "temperature_mean": 38,
        "temperature_max": 42,
        "temperature_min": 34,
        "precipitation_mm": 0,
        "wind_speed_mps": 4,
        "snowfall_cm": 0,
        "humidity": 55
    }

    try:

        result = run_smartgrid(sample_input)

        print("\n" + "=" * 55)
        print("             SMARTGRID AI REPORT")
        print("=" * 55)

        print("\nPredicted usage:", result["predicted_usage"])
        print("Risk status:", result["risk_status"])
        print("Risk probability:", result["risk_probability"], "%")
        print("Usage threshold:", result["usage_threshold"])

        print("\n========== GEMINI EXPLANATION ==========")
        print(result["explanation"])

        print("\n========== RETRIEVED SOURCES ==========")

        for index, source in enumerate(
            result["sources"], start=1
        ):
            print(f"\nSource {index}")
            print("Document:", source["document"])
            print("Page:", source["page"])
            print("Similarity:", source["similarity_score"])

        print("\n========== EXECUTION COMPLETED ==========")

    except Exception as e:

        print("\nSMARTGRID PIPELINE ERROR")
        print(type(e).__name__ + ":", str(e))
