from __future__ import annotations
import time
import json
import logging
from typing import Any, Dict, List
from dataclasses import dataclass, field, asdict

import numpy as np
from embeddings.embedder import embed_texts

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    query: str
    answer: str
    reference: str
    source_files: List[str]
    rouge_l: float = 0.0
    bert_score: float = 0.0
    faithfulness: float = 0.0
    precision_at_k: float = 0.0
    latency_total: float = 0.0
    node_timings: Dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    chunks_used: int = 0
    avg_retrieval_score: float = 0.0


def _rouge_l(hypothesis: str, reference: str) -> float:
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs_len = lcs(hyp_tokens, ref_tokens)
    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _bert_score(hypothesis: str, reference: str) -> float:
    hyp_emb = embed_texts([hypothesis])
    ref_emb = embed_texts([reference])
    hyp_norm = hyp_emb / (np.linalg.norm(hyp_emb) + 1e-10)
    ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
    return round(float(np.dot(hyp_norm.flatten(), ref_norm.flatten())), 4)


def _faithfulness(answer: str, source_chunks: List[str]) -> float:
    if not source_chunks or not answer:
        return 0.0
    answer_emb = embed_texts([answer])
    chunk_embs = embed_texts(source_chunks)
    answer_norm = answer_emb / (np.linalg.norm(answer_emb, axis=1, keepdims=True) + 1e-10)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-10)
    scores = (chunk_norms @ answer_norm.T).flatten()
    return round(float(np.max(scores)), 4)


def evaluate_result(
    result: Dict[str, Any],
    reference_answer: str = "",
    relevant_files: List[str] = None,
) -> EvalResult:
    query = result.get("query", "")
    answer = result.get("final_answer") or ""
    sources = result.get("source_files") or []
    timings = result.get("node_timings") or {}
    evaluation = result.get("evaluation") or {}
    reranked = result.get("reranked_chunks") or []
    source_contents = [c.get("content", "") for c in reranked]

    rouge = _rouge_l(answer, reference_answer) if reference_answer else 0.0
    bert = _bert_score(answer, reference_answer) if reference_answer else 0.0
    faith = _faithfulness(answer, source_contents) if source_contents else 0.0

    precision_k = 0.0
    if relevant_files and sources:
        hits = sum(1 for s in sources if any(rf in s for rf in relevant_files))
        precision_k = round(hits / len(sources), 4)

    latency = sum(timings.values())

    return EvalResult(
        query=query,
        answer=answer,
        reference=reference_answer,
        source_files=sources,
        rouge_l=rouge,
        bert_score=bert,
        faithfulness=faith,
        precision_at_k=precision_k,
        latency_total=round(latency, 3),
        node_timings=timings,
        retry_count=result.get("retry_count", 0),
        chunks_used=evaluation.get("chunks_used", 0),
        avg_retrieval_score=evaluation.get("avg_retrieval_score", 0.0),
    )


def run_eval_suite(test_cases: List[Dict], root_path: str, top_k: int = 5) -> List[EvalResult]:
    from graph.builder import build_graph
    graph = build_graph()
    results = []

    for case in test_cases:
        query = case["query"]
        reference = case.get("reference_answer", "")
        relevant_files = case.get("relevant_files", [])

        logger.info(f"[Eval] Running: {query}")

        initial_state = {
            "query": query,
            "root_path": root_path,
            "top_k": top_k,
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
        result["query"] = query

        eval_result = evaluate_result(result, reference, relevant_files)
        results.append(eval_result)
        logger.info(f"[Eval] ROUGE-L: {eval_result.rouge_l} | BERTScore: {eval_result.bert_score} | Faithfulness: {eval_result.faithfulness} | Latency: {eval_result.latency_total}s")

    return results


def print_eval_report(results: List[EvalResult]):
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    table = Table(title="DocLAMAR Evaluation Report", box=box.MARKDOWN)
    table.add_column("Query", style="cyan", max_width=30)
    table.add_column("ROUGE-L", style="green")
    table.add_column("BERTScore", style="green")
    table.add_column("Faithfulness", style="yellow")
    table.add_column("Precision@K", style="yellow")
    table.add_column("Latency (s)", style="magenta")
    table.add_column("Retries", style="red")

    for r in results:
        table.add_row(
            r.query[:30],
            str(r.rouge_l),
            str(r.bert_score),
            str(r.faithfulness),
            str(r.precision_at_k),
            str(r.latency_total),
            str(r.retry_count),
        )

    console.print(table)

    avg_rouge = round(sum(r.rouge_l for r in results) / len(results), 4)
    avg_bert = round(sum(r.bert_score for r in results) / len(results), 4)
    avg_faith = round(sum(r.faithfulness for r in results) / len(results), 4)
    avg_lat = round(sum(r.latency_total for r in results) / len(results), 3)

    console.print(f"\n[bold]Averages:[/bold] ROUGE-L={avg_rouge} | BERTScore={avg_bert} | Faithfulness={avg_faith} | Latency={avg_lat}s")
