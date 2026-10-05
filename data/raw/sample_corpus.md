# Sample Research Corpus — AI Safety & RAG Systems

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) combines a retriever with a generator.
The retriever finds relevant passages from a document corpus. The generator
(usually a large language model) produces an answer grounded in those passages.
RAG reduces hallucinations compared to asking an LLM with no external context.

## Limitations of Linear RAG Pipelines

A simple retrieve-then-generate pipeline can fail when:
1. The retrieved passages are irrelevant to the question.
2. The model invents facts that are not present in the context.
3. Multi-hop questions require combining evidence from multiple documents.
4. There is no recovery mechanism when retrieval quality is poor.

## LangGraph for Stateful Multi-Agent RAG

LangGraph lets developers build graphs of specialized nodes that share a typed
state. Conditional edges can route execution based on intermediate results.
Cycles enable recovery loops such as re-retrieving when relevance is low, or
regenerating when hallucinations are detected.

## Query Types in Research Assistants

Research questions are often classified as:
- Factual: a single fact can answer the question.
- Multi-hop: information must be combined across passages.
- Comparative: two or more entities or concepts are compared.
- Unanswerable: the corpus does not contain enough information.

## Multi-Query Retrieval

Multi-query retrieval generates several rephrasings of the user question,
retrieves documents for each rephrasing, then merges and deduplicates results.
This improves recall when wording mismatch would otherwise hide relevant text.

## Relevance Grading and Hallucination Checking

A relevance grader scores each retrieved chunk against the original question.
If average relevance falls below a threshold, the system can retry retrieval.
A hallucination checker verifies that claims in the generated answer are
supported by the retrieved context. Unsupported claims trigger regeneration.

## Evaluation Metrics for RAG Systems

Common evaluation dimensions include answer correctness, hallucination rate,
whether retry loops were triggered, and end-to-end latency. Fair comparisons
should use the same corpus, embedding model, and test questions for both the
baseline pipeline and the graph-based system.
