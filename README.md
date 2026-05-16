# AutoDrive QA Agent

An agentic AI system that answers complex automotive questions by reasoning over a private knowledge base of vehicle manuals and fault code databases — with memory, tool-use, and a RAG pipeline under the hood.

---

## What It Does

- Decodes OBD-II fault codes (P0420, P0300, etc.) with severity and causes
- Answers vehicle-specific questions from a Toyota Camry 2022 owner's manual using RAG
- Handles general automotive questions via direct LLM inference
- Remembers conversation context across multiple turns
- Runs completely offline — no OpenAI API, no cloud costs

---

## Architecture

## 🏗️ Architecture

```
User Question
      ↓
Streamlit UI (port 8501)
      ↓
FastAPI Backend (port 8000) → POST /chat
      ↓
LLM Router (llama3.2) — decides which tool
      ↓
┌─────────────────────────────────────────┐
│           │                   │         │
▼           ▼                   ▼         │
Fault    RAG Search          General      │
Code     ChromaDB            llama3.2     │
Decoder  Semantic Search     Direct       │
CSV      Top 3 Chunks                     │
Lookup        │                           │
│         llama3.2                        │
▼         Synthesizes                     │
llama3.2  Answer                          │
Explains                                  │
└─────────────────────────────────────────┘
                    ↓
         Conversation Memory Updated
                    ↓
          Response Returned to User
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | llama3.2 via Ollama (local) |
| Agent Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| PDF Parsing | pypdf |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + docker-compose |
| Public Access | ngrok |
| Language | Python 3.11 |

---

## Tools the Agent Uses

**Tool 1 — RAG Search**
Searches 1,811 embedded chunks from the Toyota Camry 2022 owner's manual using semantic similarity, retrieves top 3 chunks, and generates a grounded answer.

**Tool 2 — Fault Code Decoder**
Fast pandas lookup against an OBD-II fault codes CSV covering 48 codes across all major vehicle systems. Returns description, system, severity, and possible causes — then explains it naturally via LLM.

**Tool 3 — General Automotive Answer**
Direct LLM inference for broad automotive questions outside the manual scope.

---

## Setup & Run

### Prerequisites
- Python 3.11+
- Docker Desktop
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repo
```bash
git clone https://github.com/narendradamera23/autodrive-qa-agent.git
cd autodrive-qa-agent
```

### 2. Pull the LLM
```bash
ollama pull llama3.2
```

### 3. Set up virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 4. Run the ingestion pipeline
Open Jupyter and run `02_ingestion.ipynb` to embed the PDF and build the ChromaDB vector store.

### 5. Start with Docker
```bash
docker-compose up --build
```

### 6. Open the app
http://localhost:8501

---

## Example Questions

- `What does fault code P0420 mean?`
- `How do I check the engine oil level?`
- `Explain fault code P0300 and how serious it is`
- `What is the difference between AWD and FWD?`
- `How does the air conditioning system work?`
- `What should I do when the tire pressure warning light comes on?`

---

## Project Structure

autodrive-qa-agent/
├── agent/
│   ├── agent.py          # Custom router agent with memory
│   ├── tools.py          # RAG search, fault decoder, general QA
│   └── memory.py         # Conversation memory
├── api/
│   └── main.py           # FastAPI backend
├── app/
│   └── streamlit_app.py  # Streamlit chat UI
├── data/
│   ├── raw_docs/         # Toyota Camry PDF
│   ├── fault_codes/      # OBD-II CSV
│   └── chromadb/         # Persistent vector store
├── ingestion/
│   ├── loader.py         # PDF loader
│   └── embedder.py       # Embedding pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

---

## Author

**Narendra Damera**
- [LinkedIn](https://www.linkedin.com/in/narendradamera)
- [GitHub](https://github.com/narendradamera23)