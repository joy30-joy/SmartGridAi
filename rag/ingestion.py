
import pymupdf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents") if os.path.exists(os.path.join(BASE_DIR, "documents")) else "documents"


def load_pdfs(folder_path=DOCUMENTS_PATH):

    documents = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".pdf"):

            file_path = os.path.join(folder_path, filename)

            print(f"Processing: {filename}")

            pdf = pymupdf.open(file_path)

            for page_number, page in enumerate(pdf):

                text = page.get_text("text").strip()

                if text:
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": page_number + 1
                        }
                    })

            pdf.close()

    print(f"\nTotal pages extracted: {len(documents)}")

    return documents


if __name__ == "__main__":

    extracted_documents = load_pdfs()

    print("\nSample extracted text:\n")
    if extracted_documents:
        print(extracted_documents[0]["text"][:1000])
    else:
        print("No text extracted. Check your PDF files.")
