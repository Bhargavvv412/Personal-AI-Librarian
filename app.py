import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pypdf
import os
import io

# ------------------------ CONFIG ------------------------
st.set_page_config(page_title="📚 Personal AI Librarian", page_icon="📘", layout="wide")
st.title("📚 Personal AI Librarian: Chat with Your Documents")

# ------------------------ HELPERS ------------------------

def get_google_api_key():
    """Get Google API key for Gemini chat model."""
    api_key = st.sidebar.text_input("🔑 Enter your Google API Key:", type="password", key="api_key_input")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
    return api_key


def get_text_from_files(uploaded_files):
    """Extract text from uploaded PDF and TXT files."""
    text = ""
    for file in uploaded_files:
        try:
            if file.type == "application/pdf":
                pdf_reader = pypdf.PdfReader(io.BytesIO(file.read()))
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            elif file.type == "text/plain":
                text += file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return text


def get_text_chunks(raw_text):
    """Split long text into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(raw_text)


@st.cache_resource
def create_vector_store(text_chunks):
    """Create and cache FAISS vector store using free HuggingFace embeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()


def get_conversational_chain():
    """Set up QA chain using Gemini model and custom prompt."""
    prompt_template = """
    You are a helpful "Personal Librarian" AI.
    Answer only from the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer concisely. If answer not found, say:
    "The answer is not available in the documents you provided."
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def handle_user_query(user_question):
    """Process user's question with FAISS + Gemini."""
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("Please upload and process your documents first.")
        return

    with st.spinner("🔍 Searching your library..."):
        docs = st.session_state.vector_store.similarity_search(user_question, k=5)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]

    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ------------------------ SIDEBAR ------------------------

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = get_google_api_key()

    st.header("📂 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files:",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )

    if st.button("🚀 Process Documents"):
        if not api_key:
            st.error("Please enter your Google API Key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing your library... (first time may take longer)"):
                raw_text = get_text_from_files(uploaded_files)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = create_vector_store(text_chunks)
                    st.success("✅ Documents processed! You can now chat below.")
                else:
                    st.error("No readable text found in the uploaded files.")


# ------------------------ MAIN CHAT ------------------------

st.header("💬 Chat with Your Librarian")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_question := st.chat_input("Ask something about your documents..."):
    if not api_key:
        st.info("Please add your Google API key first in the sidebar.")
    elif "vector_store" not in st.session_state:
        st.info("Please upload and process documents first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_question)
        handle_user_query(user_question)
        with st.chat_message("assistant"):
            st.markdown(st.session_state.messages[-1]["content"])
