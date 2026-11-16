"""NiyamSetu RAG (Retrieval-Augmented Generation) Module.

A self-contained, evidence-grounded question answering module for official
Indian government notifications, policies, and regulations.

Key architectural properties:
- Consumes pre-processed clause segments from Person 1's Backend API.
- Preserves full page provenance (document_id, version_id, clause_id, page numbers,
  and character offsets).
- Implements hybrid retrieval (BM25 + Dense FAISS via Reciprocal Rank Fusion).
- Enforces strict citation verification against source text.
- Gates answering on multi-factor evidence sufficiency (graceful refusal).
"""

__version__ = "0.1.0"
__author__ = "Person 3 (Ask Feature + RAG NLP)"

