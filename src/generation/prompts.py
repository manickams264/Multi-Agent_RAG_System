"""Prompt templates used by graph nodes and the baseline chain."""

from __future__ import annotations

QUERY_ANALYSER_PROMPT = """You are a query analysis expert for a research RAG system.
Classify the user question and extract key entities.

Query types (pick exactly one):
- factual: single-fact lookup answerable from one passage
- multi-hop: needs combining information from multiple places
- comparative: compares two or more entities/concepts
- unanswerable: likely outside the corpus or impossible to answer from documents

Return ONLY valid JSON with this schema:
{{
  "query_type": "factual|multi-hop|comparative|unanswerable",
  "entities": ["entity1", "entity2"],
  "rationale": "one short sentence"
}}

User question: {query}
"""

QUERY_REWRITER_PROMPT = """You rewrite user questions into optimized retrieval queries.
Query type: {query_type}
Entities: {entities}
Original question: {query}

Rules:
- factual → produce 1 focused retrieval query
- multi-hop → produce 2-3 sub-queries covering different hops
- comparative → produce one query per side being compared (2+)
- unanswerable → still produce 1 best-effort retrieval query
- Always produce between 1 and 3 queries total for multi-query retrieval

Return ONLY valid JSON:
{{
  "rewritten_queries": ["query1", "query2"]
}}
"""

RELEVANCE_GRADER_PROMPT = """You grade whether a retrieved passage is relevant to a question.
Score from 0.0 (irrelevant) to 1.0 (highly relevant).

Question: {query}
Passage:
{passage}

Return ONLY valid JSON:
{{
  "score": 0.0,
  "reason": "short reason"
}}
"""

ANSWER_GENERATOR_PROMPT = """You are a careful research assistant.
Answer the question using ONLY the provided context.
Cite sources inline like [source: filename].
If the context is insufficient, say you cannot answer from the available documents.

Question: {query}

Context:
{context}

Answer:
"""

ANSWER_REGENERATE_PROMPT = """Your previous answer may contain unsupported claims.
Rewrite a safer answer using ONLY the context below.
Cite sources inline like [source: filename].
Do not invent facts. If unsure, say so.

Question: {query}

Previous answer:
{previous_answer}

Hallucination notes:
{hallucination_notes}

Context:
{context}

Revised answer:
"""

HALLUCINATION_CHECKER_PROMPT = """You verify whether an answer is grounded in the given context.
Flag any claim that is not supported by the context.

Question: {query}

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON:
{{
  "is_hallucinated": false,
  "unsupported_claims": [],
  "notes": "short summary"
}}
"""

BASELINE_ANSWER_PROMPT = """Answer the question using ONLY the context.
Cite sources when possible. If insufficient, say you cannot answer.

Question: {query}

Context:
{context}

Answer:
"""
