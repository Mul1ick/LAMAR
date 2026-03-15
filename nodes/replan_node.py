from __future__ import annotations
import logging
from typing import List
from schemas.state import DoclamarState

logger = logging.getLogger(__name__)

STOPWORDS = {
    "explain", "tell", "me", "about", "what", "is", "the", "a",
    "an", "how", "does", "do", "give", "show", "find", "describe"
}


def _broaden(query: str) -> List[str]:
    words = [w.lower() for w in query.split() if w.lower() not in STOPWORDS and len(w) > 2]
    return words if words else query.lower().split()[:3]


def replan_node(state: DoclamarState) -> DoclamarState:
    retry = state.get("retry_count", 0)
    logger.info(f"[ReplanNode] No results — broadening strategy (attempt {retry + 1})")

    original = state.get("routing_plan") or {}
    new_plan = {
        **original,
        "keywords": _broaden(state["query"]),
        "allowed_extensions": ["pdf", "txt", "docx"],
        "max_files": original.get("max_files", 25) + 25,
        "use_content_preview": True,
        "reasoning": "Broadened after empty retrieval."
    }

    return {
        **state,
        "routing_plan": new_plan,
        "candidate_documents": None,
        "parsed_chunks": None,
        "retrieved_chunks": None,
        "reranked_chunks": None,
        "retry_count": 1,
        "error": None,
    }
