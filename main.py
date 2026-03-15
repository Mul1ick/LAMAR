from __future__ import annotations
import argparse
import logging
import sys
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

console = Console()


def run(query: str, root_path: str, top_k: int = 5) -> dict:
    from graph.builder import build_graph

    graph = build_graph()
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

    console.print(Panel(
        f"[bold cyan]Query:[/bold cyan] {query}\n"
        f"[bold cyan]Path:[/bold cyan]  {root_path}\n"
        f"[bold cyan]Top-K:[/bold cyan] {top_k}",
        title="[bold green]DocLAMAR — Agentic Document Intelligence",
        border_style="green",
    ))

    result = graph.invoke(initial_state)

    answer = str(result.get("final_answer") or "No answer generated.")
    sources = result.get("source_files") or []
    evaluation = result.get("evaluation") or {}
    timings = result.get("node_timings") or {}
    error = result.get("error") or ""
    retries = result.get("retry_count", 0)

    if error:
        console.print(Panel(str(error), title="[bold red]Error", border_style="red"))

    console.print(Panel(answer, title="[bold green]Answer", border_style="green"))

    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for s in sources:
            console.print(f"  [cyan]•[/cyan] {s}")

    if timings:
        table = Table(title="Node Timings", box=box.SIMPLE)
        table.add_column("Node", style="cyan")
        table.add_column("Time (s)", style="yellow")
        for node, t in timings.items():
            table.add_row(node, str(t))
        console.print(table)

    if evaluation:
        console.print(f"\n[dim]Chunks used: {evaluation.get('chunks_used')} | "
                      f"Avg score: {evaluation.get('avg_retrieval_score')} | "
                      f"Retries: {retries}[/dim]")

    return result


def main():
    parser = argparse.ArgumentParser(description="DocLAMAR: Agentic Document RAG")
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--root_path", "-p", required=True)
    parser.add_argument("--top_k", "-k", type=int, default=5)
    args = parser.parse_args()
    result = run(args.query, args.root_path, args.top_k)
    if result.get("error") and not result.get("final_answer"):
        sys.exit(1)


if __name__ == "__main__":
    main()
