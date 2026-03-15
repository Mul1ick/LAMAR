from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.builder import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocLAMAR API",
    description="Agentic Document Intelligence System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class QueryRequest(BaseModel):
    query: str
    root_path: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_files: List[str]
    evaluation: Optional[Dict[str, Any]] = None
    node_timings: Optional[Dict[str, float]] = None
    retry_count: int
    error: Optional[str] = None


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.root_path.strip():
        raise HTTPException(status_code=400, detail="root_path cannot be empty.")

    graph = get_graph()

    initial_state = {
        "query": request.query,
        "root_path": request.root_path,
        "top_k": request.top_k,
        "routing_plan": None,
        "candidate_documents": None,
        "parsed_chunks": None,
        "retrieved_chunks": None,
        "reranked_chunks": None,
        "final_answer": None,
        "source_files": None,
        "evaluation": None,
        "error": None,
        "retry_count": 0,
        "node_timings": {},
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        answer=result.get("final_answer") or "No answer generated.",
        source_files=result.get("source_files") or [],
        evaluation=result.get("evaluation"),
        node_timings=result.get("node_timings"),
        retry_count=result.get("retry_count", 0),
        error=result.get("error"),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
