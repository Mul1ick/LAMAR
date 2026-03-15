from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

from schemas.state import DoclamarState
from schemas.models import DocumentSchema
from utils.llm_provider import get_llm
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

STOPWORDS = {
    "explain", "tell", "me", "about", "what", "is", "the", "a",
    "an", "how", "does", "do", "give", "show", "find", "in", "of",
    "for", "describe", "summarize", "list", "all", "any", "some"
}


def _llm_routing_plan(query: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt = f"""You are a file routing planner for a local document retrieval system.

User query: "{query}"

Return a JSON object with this exact schema:
{{
  "keywords": ["keyword1", "keyword2"],
  "allowed_extensions": ["pdf", "txt", "docx"],
  "max_files": 25,
  "use_content_preview": true,
  "reasoning": "brief explanation"
}}

Rules:
- keywords: 3-6 core concepts from the query, lowercase
- allowed_extensions: only include types likely to contain the answer
- max_files: integer between 10 and 50
- use_content_preview: true if filenames alone may not indicate relevance"""

    try:
        raw = llm.generate_json(prompt)
        data = extract_json(raw)
        assert isinstance(data.get("keywords"), list)
        assert isinstance(data.get("allowed_extensions"), list)
        return data
    except Exception as e:
        logger.warning(f"[RoutingNode] LLM plan failed: {e}. Using fallback.")
        keywords = [w for w in query.lower().split() if w not in STOPWORDS and len(w) > 2]
        return {
            "keywords": keywords or query.lower().split()[:4],
            "allowed_extensions": ["pdf", "txt", "docx"],
            "max_files": 25,
            "use_content_preview": True,
            "reasoning": "Fallback plan from query tokenization."
        }


def _route_files(root_path: str, plan: Dict[str, Any]) -> List[DocumentSchema]:
    keywords = [k.lower() for k in plan.get("keywords", [])]
    allowed_exts = [e.lower().lstrip(".") for e in plan.get("allowed_extensions", ["pdf", "txt", "docx"])]
    max_files = int(plan.get("max_files", 25))
    use_preview = bool(plan.get("use_content_preview", True))

    results: List[DocumentSchema] = []

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if len(results) >= max_files:
                return results

            ext = Path(filename).suffix.lstrip(".").lower()
            if ext not in allowed_exts:
                continue

            full_path = os.path.join(dirpath, filename)
            normalized = filename.lower().replace("_", " ").replace("-", " ")
            searchable = f"{full_path.lower()} {normalized}"
            match = any(kw in searchable for kw in keywords)

            if not match and use_preview:
                try:
                    with open(full_path, "rb") as fh:
                        preview = fh.read(2000).decode(errors="ignore").lower()
                    match = any(kw in preview for kw in keywords)
                except OSError:
                    pass

            if not match:
                continue

            size_kb = round(os.path.getsize(full_path) / 1024, 2)
            results.append(DocumentSchema(
                file_path=full_path,
                file_name=filename,
                file_type=ext,
                size_kb=size_kb,
            ))

    return results


def routing_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[RoutingNode] Starting...")

    try:
        plan = _llm_routing_plan(state["query"])
        documents = _route_files(state["root_path"], plan)
        elapsed = round(time.time() - t0, 3)
        timings = dict(state.get("node_timings") or {})
        timings["routing"] = elapsed

        logger.info(f"[RoutingNode] Found {len(documents)} document(s) in {elapsed}s")
        return {
            **state,
            "routing_plan": plan,
            "candidate_documents": [d.model_dump() for d in documents],
            "error": None if documents else "No matching documents found.",
            "node_timings": timings,
        }
    except Exception as e:
        logger.error(f"[RoutingNode] Fatal: {e}")
        return {**state, "error": str(e), "candidate_documents": [], "node_timings": state.get("node_timings") or {}}
