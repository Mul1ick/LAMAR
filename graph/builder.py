from __future__ import annotations
from langgraph.graph import StateGraph, END
from schemas.state import DoclamarState
from nodes.routing_node import routing_node
from nodes.parsing_node import parsing_node
from nodes.retrieval_node import retrieval_node
from nodes.reranking_node import reranking_node
from nodes.summarization_node import summarization_node
from nodes.replan_node import replan_node


def _after_routing(state: DoclamarState) -> str:
    if state.get("error") and not state.get("candidate_documents"):
        return "replan_node"
    if not state.get("candidate_documents"):
        return END
    return "parsing_node"


def _after_reranking(state: DoclamarState) -> str:
    if not state.get("reranked_chunks") and state.get("retry_count", 0) < 2:
        return "replan_node"
    return "summarization_node"


def _after_replan(state: DoclamarState) -> str:
    if state.get("retry_count", 0) >= 2:
        return END
    return "routing_node"


def build_graph():
    workflow = StateGraph(DoclamarState)

    workflow.add_node("routing_node", routing_node)
    workflow.add_node("parsing_node", parsing_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("reranking_node", reranking_node)
    workflow.add_node("summarization_node", summarization_node)
    workflow.add_node("replan_node", replan_node)

    workflow.set_entry_point("routing_node")

    workflow.add_conditional_edges(
        "routing_node",
        _after_routing,
        {"parsing_node": "parsing_node", "replan_node": "replan_node", END: END}
    )
    workflow.add_edge("parsing_node", "retrieval_node")
    workflow.add_edge("retrieval_node", "reranking_node")
    workflow.add_conditional_edges(
        "reranking_node",
        _after_reranking,
        {"summarization_node": "summarization_node", "replan_node": "replan_node"}
    )
    workflow.add_edge("summarization_node", END)
    workflow.add_conditional_edges(
        "replan_node",
        _after_replan,
        {"routing_node": "routing_node", END: END}
    )

    return workflow.compile()
