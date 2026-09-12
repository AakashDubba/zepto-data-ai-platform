"""
Zepto Support Assistant — FastAPI Application
==============================================
Exposes POST /ask endpoint accepting {"query": str} and returning
validated JSON with answer, sources, and confidence.
"""

import os
import sys

# Ensure support_assistant directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag_engine import run_query

app = FastAPI(
    title="Zepto Support Assistant",
    description="AI-powered customer support assistant using RAG pipeline",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request schema for /ask endpoint."""
    query: str = Field(..., min_length=1, description="Customer query text")


class QueryResponse(BaseModel):
    """Response schema for /ask endpoint."""
    answer: str
    sources: list[str]
    confidence: float


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Zepto Support Assistant"}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Process a customer query through the RAG pipeline.
    
    - Policy-related queries are routed to ChromaDB retrieval.
    - General queries receive a canned response.
    - All outputs are validated via Pydantic schema.
    """
    try:
        result = run_query(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
