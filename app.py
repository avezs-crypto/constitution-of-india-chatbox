
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Constitution of India Q&A",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Resources (cached for performance) ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_db(embeddings):
    return FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except KeyError:
            st.error("GROQ_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
            st.stop()
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=api_key
    )

try:
    emb = load_embeddings()
    database = load_faiss_db(emb)
    llm = load_llm()

    r = database.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# --- Prediction Function ---
def prediction(question):
    try:
        docs = r.invoke(question)
        context = ""
        sources = []
        
        for i, doc in enumerate(docs):
            context += doc.page_content + "\n"
            # Extract source metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                sources.append({
                    'index': i + 1,
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                })
        
        prompt = f"""You are an expert constitutional lawyer. Answer questions based on the Constitution of India in simple language that a non-lawyer can understand.

Context from the Constitution of India:
{context}

User Question: {question}

Provide a clear, concise answer citing relevant sections when applicable."""

        answer = llm.invoke(prompt)
        return answer.content, sources, docs
    except Exception as e:
        return f"Error: {str(e)}", [], []

# --- Streamlit UI ---
st.title("📜 Constitution of India Q&A Chatbot")
st.markdown("Ask questions about the Indian Constitution and get answers powered by AI")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about the Constitution of India.")
    st.write("**How it works:**")
    st.write("1. Your question is matched with relevant Constitutional sections")
    st.write("2. Those sections are sent to an AI model for accurate interpretation")
    st.write("3. You get a simple, clear answer with source citations")
    st.divider()
    st.write("Built with Streamlit + LangChain + FAISS + LLaMA 3.3")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.write(f"**Source {source['index']}:**")
                    st.write(source['content'])

# Input area
user_question = st.chat_input("Ask a question about the Constitution of India...")

if user_question:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("user"):
        st.write(user_question)
    
    # Get answer
    with st.spinner("🔍 Searching the Constitution... This may take a moment..."):
        answer, sources, raw_docs = prediction(user_question)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)
        
        # Show sources
        if sources:
            with st.expander("📚 View Sources"):
                for source in sources:
                    st.write(f"**Source {source['index']}:**")
                    st.write(source['content'])
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

