"""Compile the LangGraph multi-agent RAG graph (Tasks 2 & 4)."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import Settings, get_settings
from src.graph.edges import route_after_hallucination, route_after_relevance
from src.graph.nodes import GraphNodes
from src.graph.state import GraphState


def build_graph(
    vector_store: Chroma | None = None,
    settings: Settings | None = None,
    use_multi_query: bool = True,
    checkpointer: Any | None = None,
):
    """
    Wire nodes and conditional edges:

      START → query_analyser → query_rewriter → retriever → relevance_grader
                ↑_________________________________| (if low quality)
                                              ↓
                                       answer_generator ←──┐
                                              ↓            │
                                     hallucination_checker ┘ (if hallucinated)
                                              ↓
                                             END
    """
    settings = settings or get_settings()
    nodes = GraphNodes(
        vector_store=vector_store,
        settings=settings,
        use_multi_query=use_multi_query,
    )

    graph = StateGraph(GraphState)

    graph.add_node("query_analyser", nodes.query_analyser)
    graph.add_node("query_rewriter", nodes.query_rewriter)
    graph.add_node("retriever", nodes.retriever)
    graph.add_node("relevance_grader", nodes.relevance_grader)
    graph.add_node("answer_generator", nodes.answer_generator)
    graph.add_node("hallucination_checker", nodes.hallucination_checker)

    graph.add_edge(START, "query_analyser")
    graph.add_edge("query_analyser", "query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "relevance_grader")

    graph.add_conditional_edges(
        "relevance_grader",
        route_after_relevance,
        {
            "retriever": "retriever",
            "answer_generator": "answer_generator",
        },
    )

    graph.add_edge("answer_generator", "hallucination_checker")

    graph.add_conditional_edges(
        "hallucination_checker",
        route_after_hallucination,
        {
            "answer_generator": "answer_generator",
            "end": END,
        },
    )

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


def export_graph_mermaid(compiled_graph) -> str:
    """Return Mermaid diagram source for Task 6 visualization."""
    try:
        return compiled_graph.get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover
        return f"# Mermaid export failed: {exc}"
