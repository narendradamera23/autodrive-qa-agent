from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import get_agent

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoDrive QA Agent API",
    description="Agentic AI system for automotive technical documentation",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Agent instance (shared across requests) ───────────────────────────────
agent = get_agent()

# ── Request/Response Models ───────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    chat_history_length: int

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "app": "AutoDrive QA Agent",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = agent.invoke(request.question)
        return QueryResponse(
            question=request.question,
            answer=answer,
            chat_history_length=len(agent.chat_history)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
def reset_memory():
    agent.chat_history = []
    return {"status": "Memory cleared successfully"}