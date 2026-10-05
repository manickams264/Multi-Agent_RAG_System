# Stateful Multi-Agent RAG System with LangGraph

End-of-training project: a **stateful, multi-agent RAG** system orchestrated by **LangGraph**, with relevance recovery loops, hallucination checks, multi-query retrieval, checkpointing, and baseline evaluation.

## Architecture

```
START
  → query_analyser
  → query_rewriter
  → retriever                    ←──┐  (if relevance low & retries left)
  → relevance_grader ───────────────┘
  → answer_generator             ←──┐  (if hallucinated & retries left)
  → hallucination_checker ──────────┘
  → END
```

Labeled Mermaid diagram (conditional edges annotated): [`assets/graph.mmd`](assets/graph.mmd).  
Export a PNG to `assets/graph.png` for the notebook/README submission.

All nodes share one typed `GraphState` (`src/graph/state.py`). No side channels.

### Task coverage

| Task | What | Where |
|------|------|--------|
| 1 State schema | `GraphState` TypedDict | `src/graph/state.py` |
| 2 Graph | 6 nodes + 2 conditional edges + cycles | `src/graph/nodes.py`, `edges.py`, `builder.py` |
| 3 Contextual RAG | **Multi-query retrieval** (merge + dedupe) | `src/retrieval/retriever.py` |
| 4 Checkpointing & logging | MemorySaver + transition log + JSON trace | `src/graph/builder.py`, `logging_utils.py`, `runner.py` |
| 5 Evaluation | Baseline vs graph helpers + ≥20 questions | `src/generation/chains.py`, `tests/evaluation/`, `notebooks/` |
| 6 Visualization | Mermaid export from compiled graph | `src/graph/builder.py` → `assets/` |

## Folder structure

```
data/raw|processed|vector_store|traces
src/ingestion|indexing|retrieval|generation|graph
app/api.py, app/ui.py
notebooks/
tests/
assets/
```

## Quick start

### 1. Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your **Gemini** key:

1. Open [Google AI Studio](https://aistudio.google.com/apikey) and create a free API key  
2. Put it in `.env` as `GOOGLE_API_KEY=...`  
3. Keep `LLM_PROVIDER=gemini` (default)

Local embeddings stay on by default (`USE_LOCAL_EMBEDDINGS=true`), so you only need the Gemini chat key — no OpenAI key required.

### 2. Add corpus & index

Put mentor documents in `data/raw/` (PDF / TXT / MD / HTML). Sample docs are already included for smoke tests.

```bash
python -m src.pipeline
```

### 3. Ask a question

```python
from src.graph.runner import run_rag

result = run_rag(
    "Why do linear RAG pipelines fail?",
    save_trace=True,
    trace_name="complex_query_trace",
)
print(result["final_answer"])
```

### 4. Baseline comparison

```python
from src.generation.chains import run_baseline
print(run_baseline("Why do linear RAG pipelines fail?")["answer"])
```

### 5. API / UI

```bash
uvicorn app.api:app --reload
streamlit run app/ui.py
```

### 6. Tests (no LLM)

```bash
pytest tests/test_graph.py tests/test_retrieval.py -q
```

### 7. Graph diagram (Task 6)

```python
from src.graph.builder import build_graph, export_graph_mermaid
from pathlib import Path

g = build_graph()
Path("assets/graph.mmd").write_text(export_graph_mermaid(g), encoding="utf-8")
```

Paste the Mermaid into a renderer (or GitHub README) and export PNG to `assets/graph.png`.

## Conditional edge conditions

1. **After `relevance_grader`**: if `retrieval_quality_ok` is false → `retriever`; else → `answer_generator`.
2. **After `hallucination_checker`**: if `final_answer` empty and hallucinated → `answer_generator`; else → `END`.

Max retries: `MAX_RETRIEVAL_RETRIES` / `MAX_GENERATION_RETRIES` in `.env`.

## Submission checklist

- [ ] Mentor corpus indexed
- [ ] Graph runs with both recovery loops
- [ ] Multi-query vs plain similarity on ≥15 questions
- [ ] ≥20-question eval vs baseline in notebook
- [ ] Full state trace JSON under `data/traces/`
- [ ] Labeled graph image in `assets/` and README
- [ ] Push to GitHub

## License

Training / educational use.
