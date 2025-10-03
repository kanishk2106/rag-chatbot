import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG Chatbot")

# Keep chat history in session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    role, text = msg
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Input box for new question
if prompt := st.chat_input("Ask me anything about your docs..."):
    # Show user message
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append(("user", prompt))

    # Send request to FastAPI backend
    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",  # backend must be running
            json={"question": prompt},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer returned.")
        else:
            answer = f"Error {response.status_code}: {response.text}"
    except Exception as e:
        answer = f"❌ API connection failed: {e}"

    # Show assistant message
    st.chat_message("assistant").write(answer)
    st.session_state["messages"].append(("assistant", answer))
