"""The Ask router.

Mounted by ``backend/app/api/main.py`` under ``/api``, alongside health,
documents, search and compare — the convention ``library_search/api.py``
established and ``comparison/api.py`` followed. It reuses the existing
machinery rather than repeating it: ``api.errors.ApiError`` for the single
§12.1 error envelope, Pydantic for validation, and a dependency seam so the
routes can be tested with no database and no model.

Two endpoints, and deliberately no more:

* ``POST /api/ask`` — ask a question, get an answer with checked citations.
* ``GET /api/ask/status`` — whether Ask can answer at all, what retrieval
  components are live, and which generation provider is configured. The
  frontend reads this once on load so it can tell the user *before* they type
  that no model is configured, instead of after they wait for an answer.

**There is no endpoint that re-exposes retrieval on its own.**
``GET /api/search/clauses`` already does clause search, and a second ranked
clause endpoint would be two places defining what a clause hit looks like.
``POST /api/ask`` returns its retrieved passages inside the answer, which is
where they are useful.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, status

from backend.app.api.dependencies import Repository, get_repository
from backend.app.api.errors import ApiError
from backend.app.api.schemas import ErrorResponse
from backend.app.ask.schemas import (
    AskRequest,
    AskResponse,
    AskStatusResponse,
    CitationOut,
    PageSpanOut,
    PassageOut,
)
from backend.app.ask.service import AskService, get_ask_service
from rag_module.core.models import RAGQuery, RAGResponse

logger = logging.getLogger("niyamsetu.ask.api")

__all__ = ["ask_router", "get_ask_service_dependency"]

ask_router = APIRouter(prefix="/ask", tags=["ask"])

ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Not found"},
    422: {"model": ErrorResponse, "description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal error"},
}


def get_ask_service_dependency(
    repository: Repository = Depends(get_repository),
) -> AskService:
    """The shared Ask service. Overridden wholesale in tests."""
    return get_ask_service(repository)


@ask_router.post(
    "",
    response_model=AskResponse,
    summary="Ask a question and get an answer grounded in the stored documents",
    responses=ERROR_RESPONSES,
)
def ask(
    request: AskRequest = Body(
        ...,
        openapi_examples={
            "factoid": {
                "summary": "A question with one clear answer in the corpus",
                "value": {
                    "question": "Which bank is the Lead Bank for the Old Delhi district?"
                },
            },
            "scoped": {
                "summary": "Restricted to one document",
                "value": {
                    "question": "What are the reporting requirements?",
                    "document_id": "DOC-0001",
                },
            },
        },
    ),
    service: AskService = Depends(get_ask_service_dependency),
) -> AskResponse:
    """Retrieve, generate and verify — then return all three.

    Every citation in the response was checked against the stored clause text
    before it was returned: ``verification_status`` says which check passed,
    and a citation that failed both is dropped rather than shown. A marker the
    model invented is stripped from the answer and reported in ``warnings``.

    ``generation_mode`` is the field to read before trusting the prose:
    ``llm`` means a configured language model wrote it; ``extractive`` means no
    model is configured and the answer is assembled verbatim out of the source
    clauses. The second is never presented as the first.
    """
    question = (request.question or "").strip()
    if not question:
        raise ApiError(
            "VALIDATION_ERROR",
            "A question is required.",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"field": "question"},
        )

    try:
        query = RAGQuery(
            question=question,
            document_ids=[request.document_id] if request.document_id else None,
            version_ids=[request.version_id] if request.version_id else None,
            segmentation_method=request.method,
            retrieval_strategy="hybrid",
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise ApiError(
            "VALIDATION_ERROR",
            str(exc),
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        ) from None

    try:
        result = service.ask(query)
    except Exception as exc:  # noqa: BLE001
        # A failure in retrieval or generation must not leak a traceback to
        # the browser (§20). The full detail goes to the server log.
        logger.exception("ask: unhandled failure answering %r", question[:120])
        raise ApiError(
            "INTERNAL_ERROR",
            "The assistant could not answer that question because of an "
            "internal error. The detail is in the server log.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from None

    return _to_response(result, include_passages=request.include_passages)


@ask_router.get(
    "/status",
    response_model=AskStatusResponse,
    summary="Whether Ask can answer, and what it is configured with",
)
def ask_status(
    service: AskService = Depends(get_ask_service_dependency),
) -> AskStatusResponse:
    """Corpus size, retrieval components and generation provider.

    Read by the Ask page on load so the user is told up front when no language
    model is configured, rather than discovering it after asking. The API key
    itself is never in this payload — only whether one is present.
    """
    state = service.state()
    return AskStatusResponse(
        ready=state.ready,
        corpus=state.corpus,
        retrieval=state.retrieval,
        generation=state.generation,
        build_ms=state.build_ms,
        error=state.error,
    )


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #


def _to_response(result: RAGResponse, *, include_passages: bool) -> AskResponse:
    """Map the RAG module's response onto the API's shape.

    The mapping is where the two worlds meet, and it is the only place that
    knows both. ``rag_module`` keeps its own dataclasses and never imports
    Pydantic or FastAPI; the API keeps its public shape stable even if the
    module's internals move.
    """
    metadata: Dict[str, Any] = result.metadata or {}
    generation = metadata.get("generation") or {}
    retrieval = metadata.get("retrieval") or {}
    fusion: Dict[str, Any] = metadata.get("fusion") or {}

    titles: Dict[str, str] = {}
    corpus = metadata.get("corpus") or {}
    # Document titles ride along on the passages themselves below; the corpus
    # summary carries only counts, so nothing here needs a second lookup.

    cited_clause_ids = {c.clause_id for c in result.citations}

    # Which passages made it into the model's context, and under which marker.
    context_info = metadata.get("context") or {}
    marker_by_clause: Dict[str, str] = {}
    for marker, clause_id in (context_info.get("markers") or {}).items():
        marker_by_clause[str(clause_id)] = str(marker)

    passages: List[PassageOut] = []
    if include_passages:
        for passage in result.retrieved_passages:
            segment = passage.segment
            detail = fusion.get(passage.clause_id) or {}
            passages.append(
                PassageOut(
                    clause_id=segment.clause_id,
                    document_id=segment.document_id,
                    version_id=segment.version_id,
                    document_title=segment.metadata.get("document_title"),
                    clause_number=segment.clause_number,
                    heading=segment.heading,
                    segment_type=segment.segment_type,
                    text=segment.text,
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    pages=[
                        PageSpanOut(
                            page_number=p.page_number,
                            char_start=p.char_start,
                            char_end=p.char_end,
                        )
                        for p in segment.pages
                    ],
                    score=passage.score,
                    rank=passage.rank,
                    retrieval_method=passage.retrieval_method,
                    component_ranks={
                        str(k): int(v)
                        for k, v in (detail.get("component_ranks") or {}).items()
                    },
                    marker=marker_by_clause.get(segment.clause_id),
                    cited=segment.clause_id in cited_clause_ids,
                )
            )

    citations: List[CitationOut] = []
    for citation in result.citations:
        char_start, char_end = (
            citation.char_offset if citation.char_offset else (None, None)
        )
        citations.append(
            CitationOut(
                document_id=citation.document_id,
                version_id=citation.version_id,
                clause_id=citation.clause_id,
                document_title=_title_for(result, citation.document_id),
                clause_number=_clause_number_for(result, citation.clause_id),
                heading=_heading_for(result, citation.clause_id),
                page_start=citation.page_start,
                page_end=citation.page_end,
                page_number=citation.page_number,
                char_start=char_start,
                char_end=char_end,
                quote=citation.quote,
                verification_status=citation.verification_status,
                quote_truncated=_quote_truncated(result, citation.clause_id, citation.quote),
                marker=marker_by_clause.get(citation.clause_id),
            )
        )

    return AskResponse(
        question=result.query,
        answer=result.answer,
        status=str(metadata.get("status") or "answered"),
        sufficient_evidence=result.sufficient_evidence,
        generation_mode=str(generation.get("mode") or "none"),
        generation_provider=generation.get("provider"),
        generation_model=generation.get("model"),
        retrieval_strategy=result.retrieval_strategy,
        retrieval_components=list(retrieval.get("components") or []),
        citations=citations,
        passages=passages,
        warnings=list(metadata.get("warnings") or []),
        took_ms=float(result.latency_ms or 0.0),
        diagnostics={
            "sufficiency": metadata.get("sufficiency"),
            "context": {
                key: value
                for key, value in (metadata.get("context") or {}).items()
                if key != "markers"
            },
            "citation_report": metadata.get("citation_report"),
            "corpus": corpus,
            "generation": {
                key: generation.get(key)
                for key in ("latency_ms", "prompt_tokens", "completion_tokens")
            },
        },
    )


def _quote_truncated(result: RAGResponse, clause_id: str, quote: str) -> bool:
    """Whether the cited clause continues past the quote."""
    segment = _segment_for(result, clause_id)
    if segment is None:
        return False
    return len((segment.text or '').strip()) > len(quote)


def _segment_for(result: RAGResponse, clause_id: str):
    for passage in result.retrieved_passages:
        if passage.clause_id == clause_id:
            return passage.segment
    return None


def _title_for(result: RAGResponse, document_id: str) -> Optional[str]:
    for passage in result.retrieved_passages:
        if passage.document_id == document_id:
            title = passage.segment.metadata.get("document_title")
            if title:
                return str(title)
    return None


def _clause_number_for(result: RAGResponse, clause_id: str) -> Optional[str]:
    segment = _segment_for(result, clause_id)
    return segment.clause_number if segment else None


def _heading_for(result: RAGResponse, clause_id: str) -> Optional[str]:
    segment = _segment_for(result, clause_id)
    return segment.heading if segment else None
