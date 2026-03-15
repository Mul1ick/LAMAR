from __future__ import annotations
import time
import logging
from typing import List

from schemas.state import DoclamarState
from schemas.models import ParsedChunkSchema
from utils.llm_provider import get_llm

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000


def _build_context(chunks: List[ParsedChunkSchema]) -> str:
    parts = []
    total = 0
    for chunk in chunks:
        source = chunk.metadata.get("file_name", "unknown")
        block = f"[Source: {source}]\n{chunk.content}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 150:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def summarization_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[SummarizationNode] Generating answer...")

    raw = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    if not raw:
        return {
            **state,
            "final_answer": "I could not find relevant information in the provided documents.",
            "source_files": [],
        }

    chunks = [ParsedChunkSchema(**{k: v for k, v in c.items() if k != "score"}) for c in raw]
    context = _build_context(chunks)
    query = state["query"]

    system = """You are an expert document analysis assistant. Answer questions strictly using the provided document content.
Do NOT use external knowledge. If the documents do not contain sufficient information, say so explicitly.
Always cite the source filename when referencing specific information."""

    prompt = f"""User Question: {query}

Document Excerpts:
{context}

Instructions:
- Answer precisely and concisely
- Cite source filenames inline (e.g., "According to [filename]...")
- If multiple documents are relevant, synthesize them coherently
- If the answer is not in the documents, say: "The provided documents do not contain sufficient information to answer this question."

Answer:"""

    try:
        llm = get_llm()
        answer = llm.generate(prompt, system=system)

        source_files = list({
            c.metadata.get("file_path", "")
            for c in chunks
            if c.metadata.get("file_path")
        })
        scores = [c.score for c in chunks if c.score is not None]

        elapsed = round(time.time() - t0, 3)
        timings = dict(state.get("node_timings") or {})
        timings["summarization"] = elapsed

        evaluation = {
            "chunks_used": len(chunks),
            "sources_cited": len(source_files),
            "avg_retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "top_score": round(max(scores), 4) if scores else 0.0,
        }

        logger.info(f"[SummarizationNode] Answer generated in {elapsed}s")
        return {
            **state,
            "final_answer": answer.strip(),
            "source_files": source_files,
            "evaluation": evaluation,
            "error": None,
            "node_timings": timings,
        }
    except Exception as e:
        logger.error(f"[SummarizationNode] Error: {e}")
        return {**state, "final_answer": f"Generation failed: {e}", "source_files": [], "error": str(e)}
