from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path so 'src' is discoverable when run via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from src.config import get_settings
from src.generation.chains import run_baseline
from src.graph.builder import build_graph, export_graph_mermaid
from src.graph.runner import run_rag
from src.pipeline import ingest_and_index

NODE_ICONS = {
    "query_analyser": "🔍",
    "query_rewriter": "✍️",
    "retriever": "📚",
    "relevance_grader": "⚖️",
    "answer_generator": "🤖",
    "hallucination_checker": "🛡️",
}


def render_mermaid(code: str, height: int = 460) -> None:
    """Render a Mermaid diagram using an embedded HTML/JS component."""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <script>
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'neutral',
          flowchart: {{ useMaxWidth: false, htmlLabels: true, curve: 'linear' }}
        }});
      </script>
      <style>
        body {{
          background-color: transparent;
          margin: 0;
          display: flex;
          justify-content: center;
          align-items: center;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .mermaid {{
          text-align: center;
          width: 100%;
        }}
      </style>
    </head>
    <body>
      <div class="mermaid">
{code}
      </div>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=True)


def display_transition_timeline(transition_log: list[dict]) -> None:
    """Render an interactive chronological timeline of graph state transitions."""
    if not transition_log:
        st.info("No state transitions recorded.")
        return

    st.markdown("#### 🔄 Step-by-Step Node Execution Timeline")
    for i, step in enumerate(transition_log, start=1):
        node = step.get("node", "unknown")
        icon = NODE_ICONS.get(node, "⚙️")
        summary = step.get("summary", "")
        timestamp = step.get("timestamp", "")
        changed = step.get("changed_fields", {})

        with st.expander(f"**Step {i}:** {icon} `{node}` — {summary}", expanded=(i == len(transition_log))):
            if timestamp:
                st.caption(f"🕒 Timestamp: `{timestamp}`")
            if changed:
                st.json(changed)


def main() -> None:
    st.set_page_config(page_title="Stateful Multi-Agent RAG", layout="wide", page_icon="🤖")
    st.title("Stateful Multi-Agent RAG (LangGraph)")
    st.caption("Analyse → Rewrite → Retrieve → Grade → Generate → Check")

    # Sidebar: Document Ingestion
    with st.sidebar:
        st.header("📁 Document Ingestion")
        st.write("Upload documents to index into ChromaDB:")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt", "md", "json", "html", "htm"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, MD, JSON, HTML",
        )

        if st.button("📥 Process & Index Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one file first.")
            else:
                settings = get_settings()
                raw_dir = settings.raw_path
                raw_dir.mkdir(parents=True, exist_ok=True)

                with st.spinner(f"Saving and indexing {len(uploaded_files)} document(s)..."):
                    for uploaded_file in uploaded_files:
                        target_file = raw_dir / uploaded_file.name
                        with open(target_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    try:
                        summary = ingest_and_index(settings=settings)
                        st.success(
                            f"**Indexing Complete!**\n\n"
                            f"- Total documents: {summary['num_documents']}\n"
                            f"- Total chunks: {summary['num_chunks']}\n"
                            f"- Vector count: {summary['collection_count']}"
                        )
                    except Exception as exc:
                        st.error(f"Error during indexing: {exc}")

        # List existing corpus
        settings = get_settings()
        raw_dir = settings.raw_path
        if raw_dir.exists():
            existing = [
                f.name
                for f in raw_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
            if existing:
                with st.expander(f"📚 Current Corpus ({len(existing)} files)"):
                    for name in sorted(existing):
                        st.write(f"- `{name}`")

    # Main Tabs
    tab_query, tab_graph, tab_traces = st.tabs([
        "💬 Query & Diagnostics",
        "📊 Architecture Graph",
        "📜 Saved Traces Explorer",
    ])

    # ----------------- TAB 1: Query & Diagnostics -----------------
    with tab_query:
        query = st.text_area("Research question", height=100, placeholder="e.g. Compare factual and multi-hop questions and explain how recovery loops help RAG.")
        col1, col2 = st.columns(2)
        with col1:
            use_graph = st.checkbox("Use LangGraph multi-agent system", value=True)
        with col2:
            use_multi_query = st.checkbox("Multi-query retrieval (rewrite & dedupe)", value=True)

        if st.button("Ask Question", type="primary") and query.strip():
            with st.spinner("Processing through multi-agent graph..."):
                trace_id = f"ui_trace_{int(time.time())}"
                if use_graph:
                    result = run_rag(
                        query.strip(),
                        use_multi_query=use_multi_query,
                        save_trace=True,
                        trace_name=trace_id,
                    )
                else:
                    result = run_baseline(query.strip())

            st.subheader("💡 Answer")
            st.markdown(result.get("final_answer") or result.get("answer", ""))

            # Diagnostics & State
            if use_graph:
                st.divider()
                st.subheader("📊 Execution Diagnostics")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Query Type", str(result.get("query_type", "N/A")).capitalize())
                m2.metric("Latency", f"{result.get('latency_seconds', 0):.2f}s")
                m3.metric("Avg Relevance", f"{result.get('avg_relevance', 0):.2f}")
                m4.metric("Retrieval Retries", f"{result.get('retrieval_retry_count', 0)}")
                m5.metric("Gen Retries", f"{result.get('generation_retry_count', 0)}")
                m6.metric("Hallucinated?", "Yes ⚠️" if result.get("is_hallucinated") else "No ✅")

                # Visual Timeline
                display_transition_timeline(result.get("transition_log", []))

                # Full State & Export
                col_trace1, col_trace2 = st.columns([3, 1])
                with col_trace1:
                    with st.expander("🔍 Complete State Payload (GraphState)"):
                        st.json(result.get("full_state", {}))
                with col_trace2:
                    st.download_button(
                        label="⬇️ Download Trace JSON",
                        data=json.dumps(result.get("full_state", {}), indent=2, default=str),
                        file_name=f"{trace_id}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

    # ----------------- TAB 2: Architecture Graph -----------------
    with tab_graph:
        st.subheader("📐 LangGraph State Machine Architecture")
        st.write(
            "Visual diagram of the compiled multi-agent state graph, showing nodes and cyclical recovery edges:"
        )

        try:
            compiled_graph = build_graph()
            mermaid_source = export_graph_mermaid(compiled_graph)
            render_mermaid(mermaid_source, height=480)
        except Exception as e:
            st.error(f"Could not render Mermaid graph: {e}")

        st.markdown(
            """
            ### 🔄 Recovery Loop Conditions
            1. **Relevance Recovery Loop (`relevance_grader` ──► `retriever`)**:
               - **Condition**: Triggers when average chunk relevance score is `< 0.5` and `retrieval_retry_count < MAX_RETRIES`.
               - **Action**: Loops back to re-retrieve with alternate query perspectives.
            2. **Hallucination Recovery Loop (`hallucination_checker` ──► `answer_generator`)**:
               - **Condition**: Triggers when the checker flags unsupported claims and `generation_retry_count < MAX_RETRIES`.
               - **Action**: Feeds `hallucination_notes` back to the generator to synthesize a safer, strictly grounded answer.
            """
        )

    # ----------------- TAB 3: Saved Traces Explorer -----------------
    with tab_traces:
        st.subheader("📜 Saved Execution Traces Explorer")
        st.write("Browse and audit past state traces stored in `data/traces/`:")

        settings = get_settings()
        traces_dir = settings.traces_path
        trace_files = []
        if traces_dir.exists():
            trace_files = sorted(
                [f.name for f in traces_dir.glob("*.json")],
                reverse=True,
            )

        if not trace_files:
            st.info("No JSON traces found in `data/traces/`. Run a question with the graph to create one!")
        else:
            selected_file = st.selectbox("Select trace to inspect:", trace_files)
            if selected_file:
                trace_path = traces_dir / selected_file
                try:
                    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))

                    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                    t_col1.metric("Latency", f"{trace_data.get('latency_seconds', 'N/A')}s")
                    t_col2.metric("Retrieval Retries", trace_data.get("retrieval_retry_count", 0))
                    t_col3.metric("Generation Retries", trace_data.get("generation_retry_count", 0))
                    t_col4.metric("Hallucinated?", "Yes" if trace_data.get("is_hallucinated") else "No")

                    st.markdown(f"**Query:** `{trace_data.get('query', '')}`")
                    st.markdown(f"**Final Answer:**\n\n{trace_data.get('final_answer', '')}")

                    # Timeline from trace
                    transitions = trace_data.get("transition_log", [])
                    if transitions:
                        display_transition_timeline(transitions)

                    with st.expander("📦 Raw Trace JSON"):
                        st.json(trace_data)
                except Exception as err:
                    st.error(f"Failed to read trace file {selected_file}: {err}")


if __name__ == "__main__":
    main()
