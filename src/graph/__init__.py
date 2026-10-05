"""LangGraph multi-agent RAG package.

Import concrete modules directly to avoid heavy dependency loading at package import:

    from src.graph.state import GraphState, initial_state
    from src.graph.builder import build_graph
    from src.graph.runner import run_rag
"""

__all__ = [
    "GraphState",
    "initial_state",
    "build_graph",
    "export_graph_mermaid",
    "run_rag",
    "save_state_trace",
]


def __getattr__(name: str):
    if name in {"GraphState", "initial_state"}:
        from src.graph.state import GraphState, initial_state

        return {"GraphState": GraphState, "initial_state": initial_state}[name]
    if name in {"build_graph", "export_graph_mermaid"}:
        from src.graph.builder import build_graph, export_graph_mermaid

        return {
            "build_graph": build_graph,
            "export_graph_mermaid": export_graph_mermaid,
        }[name]
    if name in {"run_rag", "save_state_trace"}:
        from src.graph.runner import run_rag, save_state_trace

        return {"run_rag": run_rag, "save_state_trace": save_state_trace}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
