import chromadb
import pandas as pd
import requests
import json
import os
from chromadb.utils import embedding_functions

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── Shared ChromaDB collection ──────────────────────────────────────────
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path="data/chromadb")
collection = client.get_or_create_collection(
    name="automotive_docs",
    embedding_function=emb_fn
)

# ── Tool 1: RAG Search ───────────────────────────────────────────────────
def rag_search(query: str) -> str:
    """Search the automotive manual for relevant information."""
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0]
    pages = results["metadatas"][0]

    if not chunks:
        return "No relevant information found in the manual."

    context = ""
    for i, (chunk, meta) in enumerate(zip(chunks, pages)):
        context += f"[Page {meta['page']}]: {chunk}\n\n"

    prompt = f"""You are an automotive expert assistant. 
Use the following context from a Toyota Camry owner's manual to answer the question clearly and completely.
If the context doesn't fully answer the question, say so honestly.

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# ── Tool 2: Fault Code Decoder ───────────────────────────────────────────
def fault_code_decoder(code: str) -> str:
    """Look up an OBD-II fault code and return its details."""
    try:
        df = pd.read_csv("data/fault_codes/obd2_codes.csv")
        code = code.strip().upper()
        match = df[df["code"] == code]

        if match.empty:
            return f"Fault code {code} not found in the local database. Try using the RAG search tool for more information."

        row = match.iloc[0]
        return (
            f"Fault Code: {row['code']}\n"
            f"Description: {row['description']}\n"
            f"System: {row['system']}\n"
            f"Severity: {row['severity']}\n"
            f"Possible Causes: {row['possible_causes']}"
        )
    except Exception as e:
        return f"Error reading fault codes: {str(e)}"


# ── Tool 3: General LLM Fallback ─────────────────────────────────────────
def general_automotive_answer(query: str) -> str:
    """Answer general automotive questions using LLM knowledge."""
    prompt = f"""You are an expert automotive assistant with deep knowledge of 
vehicle systems, maintenance, and repair. Answer the following question 
clearly and practically.

Question: {query}

Answer:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]