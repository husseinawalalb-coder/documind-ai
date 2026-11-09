
import pymupdf


def extract_pdf_text(pdf_bytes):
    pages_data = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = len(doc)

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()

            if text.strip():
                pages_data.append({
                    "page": page_number,
                    "text": text
                })

    return pages_data, total_pages


def create_chunks(pages_data, chunk_size=500, overlap=100):
    chunks = []

    step = chunk_size - overlap

    for page_data in pages_data:
        text = page_data["text"]
        page_number = page_data["page"]

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

    return chunks
