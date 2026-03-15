from __future__ import annotations
import time
import logging
from typing import List

from schemas.state import DoclamarState
from schemas.models import ParsedChunkSchema
from vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


def retrieval_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[RetrievalNode] Building FAISS index and searching...")

    raw_chunks = state.get("parsed_chunks") or []
    if not raw_chunks:
        return {**state, "retrieved_chunks": [], "error": "No parsed chunks to index."}

    chunks = [ParsedChunkSchema(**c) for c in raw_chunks]
    query = state["query"]
    top_k = state.get("top_k", 5)

    try:
        store = FAISSStore()
        store.build(chunks)
        results = store.search(query, top_k=top_k * 3)

        retrieved = []
        for chunk, score in results:
            d = chunk.model_dump()
            d["score"] = score
            retrieved.append(d)

        elapsed = round(time.time() - t0, 3)
        timings = dict(state.get("node_timings") or {})
        timings["retrieval"] = elapsed

        logger.info(f"[RetrievalNode] Retrieved {len(retrieved)} chunk(s) in {elapsed}s")
        return {
            **state,
            "retrieved_chunks": retrieved,
            "error": None,
            "node_timings": timings,
        }
    except Exception as e:
        logger.error(f"[RetrievalNode] Error: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}
