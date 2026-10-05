"""Task 2 — Graph nodes. Each node reads GraphState and returns a partial update."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma

from src.config import Settings, get_settings
from src.generation.llm import get_llm
from src.generation.prompts import (
    ANSWER_GENERATOR_PROMPT,
    ANSWER_REGENERATE_PROMPT,
    HALLUCINATION_CHECKER_PROMPT,
    QUERY_ANALYSER_PROMPT,
    QUERY_REWRITER_PROMPT,
    RELEVANCE_GRADER_PROMPT,
)
from src.graph.logging_utils import make_transition
from src.graph.state import GraphState, RelevanceGrade
from src.graph.utils import message_to_text, parse_json_content
from src.retrieval.retriever import retrieve_multi_query, retrieve_with_scores


def _format_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no retrieved context)"
    parts = []
    for i, c in enumerate(chunks):
        src = c.get("source", "unknown")
        parts.append(f"[{i}] (source: {src}, score: {c.get('score', 0):.3f})\n{c['content']}")
    return "\n\n".join(parts)


class GraphNodes:
    """Factory holding shared dependencies (LLM, vector store, settings)."""

    def __init__(
        self,
        vector_store: Chroma | None = None,
        settings: Settings | None = None,
        use_multi_query: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = get_llm(self.settings)
        self.vector_store = vector_store
        self.use_multi_query = use_multi_query  # Task 3 enhancement toggle

    # ----- 1. Query Analyser -----
    def query_analyser(self, state: GraphState) -> dict[str, Any]:
        query = state["original_query"]
        raw = message_to_text(
            self.llm.invoke(QUERY_ANALYSER_PROMPT.format(query=query)).content
        )
        data = parse_json_content(raw)

        query_type = data.get("query_type", "factual")
        if query_type not in {"factual", "multi-hop", "comparative", "unanswerable"}:
            query_type = "factual"
        entities = data.get("entities") or []
        if not isinstance(entities, list):
            entities = [str(entities)]
        rationale = data.get("rationale", "")

        return {
            "query_type": query_type,
            "entities": [str(e) for e in entities],
            "analysis_rationale": str(rationale),
            "transition_log": [
                make_transition(
                    "query_analyser",
                    f"Classified as {query_type} with {len(entities)} entities",
                    {
                        "query_type": query_type,
                        "entities": entities,
                        "analysis_rationale": rationale,
                    },
                )
            ],
        }

    # ----- 2. Query Rewriter -----
    def query_rewriter(self, state: GraphState) -> dict[str, Any]:
        prompt = QUERY_REWRITER_PROMPT.format(
            query=state["original_query"],
            query_type=state.get("query_type", "factual"),
            entities=", ".join(state.get("entities") or []) or "(none)",
        )
        raw = message_to_text(self.llm.invoke(prompt).content)
        data = parse_json_content(raw)
        queries = data.get("rewritten_queries") or [state["original_query"]]
        if not isinstance(queries, list):
            queries = [str(queries)]
        queries = [str(q).strip() for q in queries if str(q).strip()]
        if not queries:
            queries = [state["original_query"]]
        # Cap at 3 for multi-query enhancement
        queries = queries[:3]

        return {
            "rewritten_queries": queries,
            "transition_log": [
                make_transition(
                    "query_rewriter",
                    f"Produced {len(queries)} rewritten queries",
                    {"rewritten_queries": queries},
                )
            ],
        }

    # ----- 3. Retriever (multi-query enhancement by default) -----
    def retriever(self, state: GraphState) -> dict[str, Any]:
        queries = state.get("rewritten_queries") or [state["original_query"]]
        if self.use_multi_query and len(queries) >= 1:
            chunks = retrieve_multi_query(
                queries,
                vector_store=self.vector_store,
                settings=self.settings,
            )
            mode = "multi_query"
        else:
            chunks = retrieve_with_scores(
                queries[0],
                vector_store=self.vector_store,
                settings=self.settings,
            )
            mode = "plain_similarity"

        return {
            "retrieved_chunks": chunks,
            "transition_log": [
                make_transition(
                    "retriever",
                    f"Retrieved {len(chunks)} chunks via {mode}",
                    {
                        "num_chunks": len(chunks),
                        "mode": mode,
                        "top_score": chunks[0]["score"] if chunks else None,
                    },
                )
            ],
        }

    # ----- 4. Relevance Grader -----
    def relevance_grader(self, state: GraphState) -> dict[str, Any]:
        query = state["original_query"]
        chunks = state.get("retrieved_chunks") or []
        grades: list[RelevanceGrade] = []
        top_n = min(len(chunks), max(1, self.settings.relevance_grade_top_n))

        for i, chunk in enumerate(chunks[:top_n]):
            score = float(chunk.get("score", 0.0))
            reason = "vector_score"
            if self.settings.use_llm_relevance_grader:
                try:
                    raw = message_to_text(
                        self.llm.invoke(
                            RELEVANCE_GRADER_PROMPT.format(
                                query=query, passage=chunk["content"][:1500]
                            )
                        ).content
                    )
                    data = parse_json_content(raw)
                    score = float(data.get("score", score))
                    reason = str(data.get("reason", ""))
                except Exception:
                    reason = "fallback_to_vector_score"
            score = max(0.0, min(1.0, score))
            grades.append(
                RelevanceGrade(chunk_index=i, score=score, reason=reason)
            )

        avg = (
            sum(g["score"] for g in grades) / len(grades) if grades else 0.0
        )
        quality_ok = avg >= self.settings.relevance_threshold
        retry_count = int(state.get("retrieval_retry_count") or 0)
        max_retries = int(
            state.get("max_retrieval_retries")
            or self.settings.max_retrieval_retries
        )

        # If quality is low and retries remain, bump counter (edge decides loop)
        will_retry = (not quality_ok) and (retry_count < max_retries)
        updates: dict[str, Any] = {
            "relevance_grades": grades,
            "avg_relevance": avg,
            "retrieval_quality_ok": quality_ok or not will_retry,
            "transition_log": [
                make_transition(
                    "relevance_grader",
                    f"avg_relevance={avg:.3f}, quality_ok={quality_ok}, will_retry={will_retry}",
                    {
                        "avg_relevance": avg,
                        "retrieval_quality_ok": quality_ok,
                        "will_retry": will_retry,
                        "grades": grades,
                    },
                )
            ],
        }
        if will_retry:
            updates["retrieval_retry_count"] = retry_count + 1
            # Force quality_ok False so router loops
            updates["retrieval_quality_ok"] = False
        else:
            # Exhausted retries or good quality → proceed to generate
            updates["retrieval_quality_ok"] = True

        return updates

    # ----- 5. Answer Generator -----
    def answer_generator(self, state: GraphState) -> dict[str, Any]:
        query = state["original_query"]
        chunks = state.get("retrieved_chunks") or []
        # Prefer chunks that graded well when available
        grades = state.get("relevance_grades") or []
        if grades:
            keep_idx = {
                g["chunk_index"]
                for g in grades
                if g.get("score", 0) >= self.settings.relevance_threshold * 0.8
            }
            filtered = [
                c for i, c in enumerate(chunks) if i in keep_idx
            ] or chunks
        else:
            filtered = chunks

        context = _format_context(filtered)
        regenerating = int(state.get("generation_retry_count") or 0) > 0 and state.get(
            "answer_draft"
        )

        if regenerating:
            prompt = ANSWER_REGENERATE_PROMPT.format(
                query=query,
                previous_answer=state.get("answer_draft", ""),
                hallucination_notes=state.get("hallucination_notes", ""),
                context=context,
            )
        else:
            prompt = ANSWER_GENERATOR_PROMPT.format(query=query, context=context)

        answer_text = message_to_text(self.llm.invoke(prompt).content)

        return {
            "answer_draft": answer_text,
            "final_answer": "",  # only set when hallucination check passes
            "transition_log": [
                make_transition(
                    "answer_generator",
                    f"Generated answer ({len(answer_text)} chars), regenerate={bool(regenerating)}",
                    {"answer_preview": answer_text[:200]},
                )
            ],
        }

    # ----- 6. Hallucination Checker -----
    def hallucination_checker(self, state: GraphState) -> dict[str, Any]:
        query = state["original_query"]
        answer = state.get("answer_draft") or ""
        context = _format_context(state.get("retrieved_chunks") or [])

        raw = message_to_text(
            self.llm.invoke(
                HALLUCINATION_CHECKER_PROMPT.format(
                    query=query, context=context, answer=answer
                )
            ).content
        )
        data = parse_json_content(raw)

        is_hallucinated = bool(data.get("is_hallucinated", False))
        claims = data.get("unsupported_claims") or []
        if not isinstance(claims, list):
            claims = [str(claims)]
        notes = str(data.get("notes", ""))

        retry_count = int(state.get("generation_retry_count") or 0)
        max_retries = int(
            state.get("max_generation_retries")
            or self.settings.max_generation_retries
        )
        will_retry = is_hallucinated and (retry_count < max_retries)

        updates: dict[str, Any] = {
            "is_hallucinated": is_hallucinated,
            "unsupported_claims": [str(c) for c in claims],
            "hallucination_notes": notes,
            "transition_log": [
                make_transition(
                    "hallucination_checker",
                    f"is_hallucinated={is_hallucinated}, will_retry={will_retry}",
                    {
                        "is_hallucinated": is_hallucinated,
                        "unsupported_claims": claims,
                        "will_retry": will_retry,
                    },
                )
            ],
        }

        if will_retry:
            updates["generation_retry_count"] = retry_count + 1
            updates["final_answer"] = ""
        else:
            # Accept answer (clean, or retries exhausted)
            updates["final_answer"] = answer
            if is_hallucinated:
                updates["hallucination_notes"] = (
                    notes + " | Retries exhausted; returning best-effort answer."
                ).strip(" |")

        return updates
