---
title: RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.31.0
app_file: ui.py
pinned: false
license: mit
---

# rag-chatbot

A minimal **Retrieval-Augmented Generation (RAG)** chatbot built with Python.  
It ingests documents (PDF, Markdown, TXT), splits them into chunks, and lays the groundwork for retrieval-based question answering.

This project demonstrates:
- Clean Python scripting for document processing, embedding, and vector store integration through LangChain
- Modular, testable code
- Future-ready design for integration with LLMs via LangChain

---

##  Features

- **Document ingestion**: Reads PDF, Markdown, and TXT files
- **Chunking**: Splits documents into fixed-size chunks for indexing along with overlap
- **Test coverage**: Basic Pytest suite to ensure ingestion works
- **Extensible**: Designed to add embeddings, vector stores, and retrievers later
- **Docker-ready**: Optional containerization for consistent environment

---

##  Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.11** | Core language |
| **langchain-text-splitters** | Efficient text chunking |
| **pypdf** | PDF text extraction |
| **pytest** | Testing framework |
| **Docker** | Portable, reproducible environment |

---

##  Project Structure

**Current**
rag-chatbot/
```
├── README.md # Project documentation
├── ingest.py # Ingestion + chunking script
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files & folders
├── data/ # Sample documents (empty initially)
│ └── .gitkeep
```
**Planned**
rag-chatbot/
```
├── tests/ # Unit tests
│ └── test_ingest.py # Basic ingestion test
├── src/ # Core RAG logic
│ ├── embeddings.py # Embedding generation
│ ├── vector_store.py # Vector DB integration
│ └── qa.py # Retriever + QA chain
├── notebooks/ # Experiments / EDA
└── Dockerfile # Container definition
```
Below is the link to the website : https://huggingface.co/spaces/kanishk2106/sep7-ka

