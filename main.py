from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# Import your RAG function and defaults
from src.qa import (
    answer_question,
    DEFAULT_PERSIST,
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
)

# Create FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Define request schema
class QueryRequest(BaseModel):
    question: str

# Define POST endpoint
@app.post("/ask")
async def ask(req: QueryRequest):
    try:
        result = answer_question(
            question=req.question,
            persist_dir=Path(DEFAULT_PERSIST),
            collection=DEFAULT_COLLECTION,
            embed_model=DEFAULT_EMBED_MODEL,
            llm_model=DEFAULT_LLM_MODEL,
            k=4,
        )
        return result
    except Exception as e:
        return {"error": str(e)}
