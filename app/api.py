from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.generation.chains import run_baseline
from src.graph.runner import run_rag

app = FastAPI(
    title="Stateful Multi-Agent RAG",
    description="LangGraph RAG with relevance + hallucination recovery loops",
    version="0.1.0",
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_multi_query: bool = True
    save_trace: bool = False
    baseline: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    if req.baseline:
        return run_baseline(req.query)
    return run_rag(
        req.query,
        use_multi_query=req.use_multi_query,
        save_trace=req.save_trace,
        trace_name="api_trace",
    )
