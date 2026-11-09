from pdf_utils import extract_pdf_text, create_chunks
import hashlib
import re
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import pymupdf





@st.cache_data
def generate_embeddings(texts):
    return embedding_model.encode(
        texts,
        show_progress_bar=False
    )


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="centered"
)

st.title("📄 DocuMind AI")

st.write("Ask questions about your PDF documents using AI.")

st.success("Welcome to DocuMind AI!")


def validate_citations(answer, relevant_chunks):

    # Extract cited page numbers from the answer
    cited_pages = re.findall(
    r"[\[(]Page\s+(\d+)[\])]",
    answer,
    flags=re.IGNORECASE
)

    # Get valid page numbers from retrieved chunks
    valid_pages = {
        chunk["page"] for chunk in relevant_chunks
    }

    # Find citations that are not in the retrieved context
    invalid_pages = []

    for page in cited_pages:
        if int(page) not in valid_pages:
            invalid_pages.append(int(page))

    return cited_pages, invalid_pages


# PDF Upload Section
st.subheader("Upload Your Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

pages_data = []

if uploaded_file is not None:
    st.success(f"Uploaded successfully: {uploaded_file.name}")

    pdf_bytes = uploaded_file.getvalue()

    pages_data, total_pages = st.cache_data(extract_pdf_text)(pdf_bytes)

    st.write(f"Total pages: {total_pages}")
    st.write(f"Pages containing text: {len(pages_data)}")


if not pages_data:
    st.warning(
        "This PDF contains no extractable text. "
        "Please upload a text-based PDF."
    )

    if pages_data:
        st.text_area(
            "First Page Text",
            pages_data[0]["text"],
            height=200
        )

    
# Text Chunking
chunks = create_chunks(pages_data)

st.write(f"Total chunks created: {len(chunks)}")
    
    # Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Generate embeddings
if chunks:
    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(chunk_texts)
    

    st.success("Embeddings generated successfully!")
    st.write(f"Embedding shape: {embeddings.shape}")
    
    
    # Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db_streamlit")

# Get or create a collection
if uploaded_file is not None:
    pdf_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    collection_name = f"documind_{pdf_id}"

    collection = client.get_or_create_collection(
        name=collection_name
    )
else:
    collection = None

# Prepare chunk information
ids = [f"chunk_{i}" for i in range(len(chunks))]
documents = [chunk["text"] for chunk in chunks]
metadatas = [{"page": chunk["page"]} for chunk in chunks]


# Store embeddings and chunks only when needed
if chunks and collection is not None:

    if collection.count() != len(chunks):
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        st.success("Chunks stored successfully in ChromaDB!")

    else:
        st.info("PDF chunks already stored. Reusing database.")

    st.write(f"Total stored chunks: {collection.count()}")
    
    
st.divider()
    
 # Question Section
st.subheader("Ask a Question")

# Initialise Session State
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

question = st.text_input(
    "What would you like to know about your PDF?",
    placeholder="e.g. What is inherited family trauma?"
)

if st.button("Ask DocuMind AI"):

    st.session_state.last_question = question

    if not question.strip():
        st.warning("Please enter a question first.")

    elif uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif not chunks:
        st.warning(
        "This PDF contains no extractable text. "
        "Please upload a text-based PDF."
    )

    elif collection.count() == 0:
        st.warning("No text chunks were found in the document.")

    else:
        # Convert question into an embedding
        question_embedding = embedding_model.encode(
            question
        ).tolist()

        # Retrieve 10 candidate chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(10, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # Filter relevant chunks
        relevant_chunks = []

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if distance <= 1.2:
                relevant_chunks.append({
                    "text": text,
                    "page": metadata["page"],
                    "distance": distance
                })

        # Keep the 3 closest chunks
        relevant_chunks = relevant_chunks[:3]

        st.write(
            f"Relevant chunks found: {len(relevant_chunks)}"
        )

        # Generate the AI answer
        st.subheader("AI Answer")

        if not relevant_chunks:
            st.warning(
                "I could not find the answer in the uploaded document."
            )

        else:
            # Prepare context
            context_parts = []

            for chunk in relevant_chunks:
                context_parts.append(
                    f"Page {chunk['page']}:\n{chunk['text']}"
                )

            context = "\n\n".join(context_parts)

            # Create the AI prompt
            prompt = f"""
You are DocuMind AI, a document question-answering assistant.

RULES:
1. Answer using ONLY the provided PDF context.
2. Do NOT use outside knowledge or invent information.
3. If the context does not contain enough information,
   respond exactly:
   "I could not find the answer in the uploaded document."
4. Keep your answer clear, concise, and relevant.
5. Every factual sentence must include a page citation
   in the format [Page X].
6. Only cite page numbers appearing in the context.
7. Never invent page numbers or information.

CONTEXT:
{context}

QUESTION:
{question}

IMPORTANT:
Include [Page X] citations in your answer.
If the answer is not in the context, use the exact
fallback response without citations.

ANSWER (with page citations):
"""

            try:
                with st.spinner("DocuMind AI is thinking..."):
                    response = ollama.chat(
                        model="llama3.2:1b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        options={
                            "num_ctx": 2048,
                            "temperature": 0
                        }
                    )

                answer = response["message"]["content"]

                # Display the answer
                st.write(answer)

                # Validate citations
                cited_pages, invalid_pages = validate_citations(
                    answer,
                    relevant_chunks
                )

                if invalid_pages:
                    st.error(
                        f"Invalid page citations detected: {invalid_pages}"
                    )

                elif cited_pages:
                    st.success(
                        "All cited pages exist in the retrieved context."
                    )

                else:
                    st.warning(
                        "No page citations were found in the answer."
                    )

                # Display sources
                st.subheader("Sources")

                for i, chunk in enumerate(
                    relevant_chunks,
                    start=1
                ):
                    with st.expander(
                        f"Source {i} - Page {chunk['page']}"
                    ):
                        st.write(chunk["text"])
                        st.caption(
                            f"Retrieval distance: {chunk['distance']:.3f}"
                        )

            except Exception as e:
                st.error(
                    "Unable to generate an answer. "
                    "Please check that Ollama is running."
                )
                st.caption(f"Technical details: {e}")


# Display last question
if st.session_state.last_question:
    st.caption(
        f"Last question asked: {st.session_state.last_question}"
    )

# Application footer
st.divider()

st.caption(
    "DocuMind AI | Powered by Streamlit, "
    "Sentence Transformers, ChromaDB and Ollama"
)
