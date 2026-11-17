"""Ask: the RAG question-answering feature, mounted into the existing API.

This package is the *adapter* between the backend and ``rag_module/``. It owns
the two things the RAG module deliberately does not:

* where the corpus comes from (``corpus.py`` — the live database, through the
  same repository every other endpoint reads);
* how long the index lives (``service.py`` — one process-wide index, rebuilt
  when the corpus fingerprint moves).

The retrieval, generation and citation logic all stay in ``rag_module``, which
imports nothing from ``backend``.
"""

from backend.app.ask.service import AskService, get_ask_service, reset_ask_service

__all__ = ["AskService", "get_ask_service", "reset_ask_service"]
