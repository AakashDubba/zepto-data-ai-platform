# Support Assistant Module

## Overview
AI-powered customer support assistant using a Retrieval-Augmented Generation (RAG) pipeline with LangGraph, ChromaDB, and FastAPI.

## Architecture

### RAG Pipeline Flow
```
User Query
    │
    ▼
┌─────────────────┐
│ classify_intent  │  ← Node 1: Keyword-based intent detection
│ (LangGraph Node) │     Keywords: delivery, return, refund, membership,
└────────┬────────┘     tracking, cancel, gift card, support hours
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
policy_q    general_q    ← Conditional Edge (intent routing)
    │          │
    ▼          ▼
┌──────────┐  ┌──────────────┐
│retrieve_ │  │ direct_answer │  ← Node 3: Canned response
│and_answer│  │              │     "I can only answer questions
│(Node 2)  │  │              │      about Zepto policies right now."
└──────┬───┘  └──────┬───────┘
       │             │
       ▼             ▼
  ChromaDB      Empty sources
  top-3 cosine  confidence=1.0
  similarity
       │
       ▼
  Mock answer:
  f"Based on the retrieved context: {top_chunk_snippet}"
       │
       ▼
┌─────────────────┐
│ Pydantic Schema  │  ← Validation: answer (str), sources (list[str]),
│ Validation       │     confidence (float). Retry up to 2× on failure.
└────────┬────────┘
         │
         ▼
    JSON Response
```

### Components
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Local text embeddings |
| Vector DB | ChromaDB (persistent) | Document storage & cosine similarity search |
| Orchestration | LangGraph `StateGraph` | 3-node workflow with conditional routing |
| API | FastAPI + Uvicorn | REST endpoint at `POST /ask` |
| Validation | Pydantic `BaseModel` | Schema enforcement on all outputs |
| Containerization | Docker | Isolated deployment |

## MOCK_LLM Branching

| Mode | Env Var | Behavior |
|------|---------|----------|
| **Mock (default)** | `MOCK_LLM=1` or unset | Keyword-based intent, ChromaDB retrieval runs live, canned answer format |
| **Real LLM** | `MOCK_LLM=0` | Same routing + live ChromaDB retrieval, structured prompt sent to LLM with ROLE→CONTEXT→TASK→FORMAT→LENGTH skeleton, negative constraint, and few-shot example |

### Structured Prompt (Real LLM Mode)
The prompt follows the **ROLE → CONTEXT → TASK → FORMAT → LENGTH** skeleton:
- **ROLE**: "You are a Zepto customer support assistant..."
- **CONTEXT**: Retrieved chunks from ChromaDB
- **TASK**: Answer using ONLY the provided context
- **Negative constraint**: "Do not answer using information not present in the provided context."
- **FORMAT**: JSON with answer, sources, confidence
- **LENGTH**: 2-4 sentences maximum
- **Few-shot example**: Concrete delivery fee Q&A example included

## LangGraph State Schema
```python
class RAGState(TypedDict):
    query: str              # User's input query
    intent: str             # "policy_question" or "general_question"
    retrieved_docs: list    # Top-3 ChromaDB results
    answer: str             # Generated answer
    sources: list[str]      # Source doc IDs (e.g., ["doc_01", "doc_02"])
    confidence: float       # 0.0 to 1.0
```

## Pydantic Validation
All outputs are validated against:
```python
class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float  # 0.0 to 1.0
```
On validation failure, retry with corrective prompt up to 2 additional times before returning a structured error.

## Policy Documents
8 `.txt` files in `docs/` covering:
| Doc | Topic |
|-----|-------|
| doc_01 | Delivery Policy |
| doc_02 | Return & Refund Policy |
| doc_03 | Zepto Pass Membership |
| doc_04 | Order Tracking |
| doc_05 | Order Cancellation |
| doc_06 | Gift Cards |
| doc_07 | Support Hours & Channels |
| doc_08 | Payment Methods & Security |

## Installation & Usage

### Local Development
```bash
cd support_assistant
pip install -r requirements.txt

# Step 1: Ingest documents into ChromaDB
python ingest.py

# Step 2: Start the API server
MOCK_LLM=1 uvicorn main:app --host 0.0.0.0 --port 7860

# Step 3: Test the API
python test_api.py
```

### Docker
```bash
cd support_assistant
docker build -t zepto-support .
docker run -d -p 7860:7860 --name zepto-test zepto-support
# Test: curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "delivery policy"}'
docker stop zepto-test && docker rm zepto-test
```

## API Endpoint

### `POST /ask`
**Request**:
```json
{"query": "What is Zepto's delivery fee?"}
```

**Response (policy query)**:
```json
{
  "answer": "Based on the retrieved context: Zepto offers ultra-fast delivery...",
  "sources": ["doc_01"],
  "confidence": 0.85
}
```

**Response (general query)**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
