import streamlit as st
import subprocess
from pathlib import Path
from src.qa import answer_question  # ✅ import your RAG function

# Path to Chroma DB
chroma_path = Path(".chroma")

# Auto-build Chroma DB if missing
if not chroma_path.exists():
    st.info("⚡ Building Chroma DB for the first time...")
    subprocess.run([
        "python", "src/embeddings.py",
        "--input", "data/chunks.jsonl",
        "--persist-dir", ".chroma",
        "--collection", "rag-chatbot"
    ], check=True)

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG Chatbot")

# Keep chat history in session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for role, text in st.session_state["messages"]:
    st.chat_message(role).write(text)

# Input box for new question
if prompt := st.chat_input("Ask me anything about your docs..."):
    # Show user message
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append(("user", prompt))

    # Call RAG directly instead of FastAPI
    try:
        result = answer_question(prompt)
        answer = result.get("answer", "⚠️ No answer returned.")
    except Exception as e:
        answer = f"❌ Error: {e}"

    # Show assistant message
    st.chat_message("assistant").write(answer)
    st.session_state["messages"].append(("assistant", answer))
