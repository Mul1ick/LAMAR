import os
from typing import List

from doclamar.schemas.parsed_chunk import ParsedChunkSchema
from doclamar.schemas.document import DocumentSchema

from doclamar.parsing.parse_planner import ParsePlanner
from doclamar.parsing.selective_parser import SelectiveParser

from doclamar.vectorstore.faiss_store import FaissSemanticStore
from doclamar.reranking.semantic_reranker import SemanticReRanker

from doclamar.llm.llm_provider import get_llm


def run_pipeline(
    query: str,
    root_path: str,
    top_k: int = 5,
) -> str:
    """
    End-to-end document analysis pipeline with FAISS-backed
    semantic memory and grounded LLM synthesis.
    """

    # -----------------------------
    # 1️⃣ Parse documents
    # -----------------------------
    parse_planner = ParsePlanner()
    parser = SelectiveParser()

    all_chunks: List[ParsedChunkSchema] = []

    for root, _, files in os.walk(root_path):
        for file in files:
            ext = file.split(".")[-1].lower()
            if ext not in {"pdf", "txt"}:
                continue

            full_path = os.path.join(root, file)

            doc = DocumentSchema(
                file_path=full_path,
                file_name=file,
                file_type=ext,
                size_kb=round(os.path.getsize(full_path) / 1024, 2),
            )

            plan = parse_planner.plan(query, doc)
            chunks = parser.parse(doc, plan)
            all_chunks.extend(chunks)

    if not all_chunks:
        return "No parsable content found in the document collection."

    # -----------------------------
    # 2️⃣ Index chunks (FAISS)
    # -----------------------------
    store = FaissSemanticStore()
    store.index_chunks(all_chunks)

    # -----------------------------
    # 3️⃣ Semantic retrieval
    # -----------------------------
    retrieved_chunks = store.search(query, top_k=top_k * 2)

    if not retrieved_chunks:
        return "No relevant content found via semantic retrieval."

    # -----------------------------
    # 4️⃣ Semantic re-ranking
    # -----------------------------
    reranker = SemanticReRanker()
    ranked_chunks = reranker.rank(query, retrieved_chunks, top_k=top_k)

    if not ranked_chunks:
        return "Retrieved content was not relevant to the query."

    # -----------------------------
    # 5️⃣ Final grounded synthesis
    # -----------------------------
    llm = get_llm()

    MAX_CHARS = 3000  # Safe for Phi-3 / 4k context

    context_parts = []
    total_chars = 0

    for chunk in ranked_chunks:
        block = (
            f"Source: {chunk.metadata.get('file_name')}\n"
            f"{chunk.content}"
        )

        if total_chars + len(block) > MAX_CHARS:
            break

        context_parts.append(block)
        total_chars += len(block)

    context = "\n\n".join(context_parts)

    final_prompt = f"""
You are an intelligent document analysis system.

Answer the user's question strictly using the extracted document content.
Do NOT use external knowledge.

User query:
{query}

Document excerpts:
{context}

Provide a precise, document-grounded answer.
"""

    answer = llm.generate(final_prompt)
    return answer.strip()
