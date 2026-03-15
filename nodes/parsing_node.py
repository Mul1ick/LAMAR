from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

from schemas.state import DoclamarState
from schemas.models import DocumentSchema, ParsedChunkSchema
from utils.llm_provider import get_llm
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def _plan_parse(query: str, doc: DocumentSchema) -> Dict[str, Any]:
    llm = get_llm()
    prompt = f"""You are a document parsing planner.

Query: "{query}"
Document: {doc.file_name} ({doc.file_type}, {doc.size_kb} KB)

Return JSON:
{{
  "parse_full": true,
  "max_chars": 15000,
  "reason": "brief reason"
}}

Rules: if size > 500KB set parse_full false with max_chars 8000, else parse_full true."""

    try:
        raw = llm.generate_json(prompt)
        data = extract_json(raw)
        assert isinstance(data.get("parse_full"), bool)
        return data
    except Exception:
        return {"parse_full": True, "max_chars": 15000, "reason": "Fallback: full parse."}


def _read_pdf(path: str, max_chars: int, parse_full: bool) -> str:
    import pypdf
    text = ""
    with open(path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        for page in reader.pages:
            text += (page.extract_text() or "")
            if not parse_full and len(text) >= max_chars:
                break
    return text.strip()


def _read_txt(path: str, max_chars: int, parse_full: bool) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read() if parse_full else fh.read(max_chars)


def _read_docx(path: str, max_chars: int, parse_full: bool) -> str:
    from docx import Document
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return text if parse_full else text[:max_chars]


def _read_document(doc: DocumentSchema, parse_full: bool, max_chars: int) -> str:
    ext = doc.file_type.lower()
    try:
        if ext == "pdf":
            return _read_pdf(doc.file_path, max_chars, parse_full)
        elif ext == "txt":
            return _read_txt(doc.file_path, max_chars, parse_full)
        elif ext == "docx":
            return _read_docx(doc.file_path, max_chars, parse_full)
        else:
            logger.warning(f"[ParsingNode] Unsupported type: {ext}")
            return ""
    except Exception as e:
        logger.error(f"[ParsingNode] Failed reading {doc.file_name}: {e}")
        return ""


def _chunk_text(text: str, doc: DocumentSchema) -> List[ParsedChunkSchema]:
    chunks = []
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    idx = 0
    for i in range(0, len(text), stride):
        chunk_text = text[i: i + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        chunks.append(ParsedChunkSchema(
            content=chunk_text,
            metadata={
                "file_path": doc.file_path,
                "file_name": doc.file_name,
                "file_type": doc.file_type,
                "chunk_index": str(idx),
                "chunk_start": str(i),
            }
        ))
        idx += 1
    return chunks


def parsing_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[ParsingNode] Starting...")

    candidates = state.get("candidate_documents") or []
    if not candidates:
        return {**state, "parsed_chunks": [], "error": "No candidate documents to parse."}

    all_chunks: List[Dict] = []

    for doc_dict in candidates:
        doc = DocumentSchema(**doc_dict)
        plan = _plan_parse(state["query"], doc)
        text = _read_document(doc, plan["parse_full"], plan["max_chars"])
        if not text:
            continue
        chunks = _chunk_text(text, doc)
        logger.info(f"[ParsingNode] {doc.file_name} → {len(chunks)} chunk(s)")
        all_chunks.extend([c.model_dump() for c in chunks])

    elapsed = round(time.time() - t0, 3)
    timings = dict(state.get("node_timings") or {})
    timings["parsing"] = elapsed

    logger.info(f"[ParsingNode] Total chunks: {len(all_chunks)} in {elapsed}s")
    return {
        **state,
        "parsed_chunks": all_chunks,
        "error": None if all_chunks else "Parsing produced no text chunks.",
        "node_timings": timings,
    }
