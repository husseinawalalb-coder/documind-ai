
# DocuMind AI 📄

An AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF documents
- Extract and split PDF text into chunks
- Generate semantic embeddings
- Store document embeddings in ChromaDB
- Retrieve relevant document sections
- Generate answers using a local Ollama LLM
- Display source pages and validate page citations
- Cache PDF processing and embeddings
- Handle empty or unreadable PDF documents

## Technologies

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Ollama (Llama 3.2 1B)

## How It Works

1. User uploads a PDF.
2. PyMuPDF extracts the text.
3. The text is divided into overlapping chunks.
4. Sentence Transformers generates embeddings.
5. ChromaDB stores and retrieves relevant chunks.
6. Ollama generates an answer from the retrieved context.
7. Streamlit displays the answer and sources.

## Installation

Install Python and Ollama.

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd documind-ai
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Ollama model:

```bash
ollama pull llama3.2:1b
```

Run the application:

```bash
streamlit run app.py
```

Open http://localhost:8501.

## Limitations

- Scanned PDFs without extractable text require OCR, which is not currently implemented.
- The local LLM may occasionally omit or generate incorrect citations.
- Citation validation checks whether referenced pages were retrieved, not whether the claims are factually supported.
- Ollama must be installed and running locally.

## Future Improvements

- Support multiple PDF documents
- Add chat history
- Improve citation accuracy
- Add OCR for scanned PDFs
- Deploy with a hosted LLM

## Author

Computer Science / AI Portfolio Project
