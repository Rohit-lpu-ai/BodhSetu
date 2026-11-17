"""Pydantic models for the Ask endpoints.

These reuse the project's existing conventions rather than inventing a second
response shape: identifiers are the ``public_id`` strings (``DOC-0001``,
``VER-0002``, ``CLS-0000166``), page provenance is ``page_start``/``page_end``
plus a per-page character range, and a citation carries the same fields the
Compare module's evidence does — so the frontend's existing deep link,
``document.html?document_id=…&version_id=…&clause_id=…``, works on an Ask
citation with no new routing.

The response deliberately separates four things the UI must be able to tell
apart:

* ``answer`` — the prose.
* ``citations`` — only the sources the answer actually used, each with a
  verification status.
* ``passages`` — everything retrieval returned, cited or not, so a reader can
  see what the assistant looked at and not only what it chose.
* ``warnings`` / ``status`` / ``generation.mode`` — how much to trust it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "AskRequest",
    "AskResponse",
    "AskStatusResponse",
    "CitationOut",
    "PassageOut",
    "PageSpanOut",
]


class AskRequest(BaseModel):
    """One question."""

    question: str = Field(
        ...,
        description="A natural-language question about the stored policy documents.",
        examples=["Which bank is the Lead Bank for the Old Delhi district?"],
    )
    document_id: Optional[str] = Field(
        None, description="Restrict the search to one document, e.g. DOC-0001."
    )
    version_id: Optional[str] = Field(
        None, description="Restrict the search to one version, e.g. VER-0002."
    )
    top_k: int = Field(
        6, ge=1, le=20, description="How many passages to retrieve as evidence."
    )
    method: str = Field(
        "clause_aware",
        pattern="^(clause_aware|paragraph)$",
        description="Which stored segmentation to answer from.",
    )
    include_passages: bool = Field(
        True,
        description="Return the full retrieved passages, not only the citations.",
    )


class PageSpanOut(BaseModel):
    """One page a segment touches, with the character range on that page."""

    page_number: int
    char_start: int
    char_end: int


class CitationOut(BaseModel):
    """A source the answer actually used, with its verification status."""

    document_id: str
    version_id: str
    clause_id: str
    document_title: Optional[str] = None
    clause_number: Optional[str] = None
    heading: Optional[str] = None
    page_start: int
    page_end: int
    page_number: Optional[int] = Field(
        None,
        description=(
            "The page the character offsets index into. Null for a clause "
            "spanning pages, because each page's offsets start at 0 and there "
            "is no continuous range across the break."
        ),
    )
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    quote: str = Field(
        ...,
        description="A verbatim excerpt of the cited clause's stored text.",
    )
    verification_status: str = Field(
        ...,
        description=(
            "STRICT_VERBATIM — the quote is a literal substring of the stored "
            "clause. AUDITED_SPAN_MATCH — it matches once whitespace is "
            "normalised, which is what PDF extraction damage requires. "
            "REJECTED citations are never returned."
        ),
    )
    quote_truncated: bool = Field(
        False,
        description=(
            "True when the clause is longer than the quote. The quote itself "
            "stays a verbatim substring — no ellipsis is added to it, because "
            "that would break the verbatim check — so a UI wanting to show "
            "one appends its own."
        ),
    )
    marker: Optional[str] = Field(
        None, description="The [S1]-style marker this citation carries in the answer."
    )


class PassageOut(BaseModel):
    """One retrieved passage, cited or not."""

    clause_id: str
    document_id: str
    version_id: str
    document_title: Optional[str] = None
    clause_number: Optional[str] = None
    heading: Optional[str] = None
    segment_type: Optional[str] = None
    text: str
    page_start: int
    page_end: int
    pages: List[PageSpanOut] = Field(default_factory=list)
    score: float = Field(..., description="Fused RRF score; not a probability.")
    rank: int
    retrieval_method: str
    component_ranks: Dict[str, int] = Field(
        default_factory=dict,
        description="Where each retriever ranked this clause before fusion.",
    )
    marker: Optional[str] = Field(
        None, description="Set when this passage was placed in the model's context."
    )
    cited: bool = False


class AskResponse(BaseModel):
    """The answer, its evidence, and how much to trust it."""

    question: str
    answer: str
    status: str = Field(
        ...,
        description=(
            "answered | insufficient_evidence | empty_corpus | no_evidence | "
            "provider_unavailable | generation_timeout | invalid_model_response "
            "| generation_failed | question_too_long"
        ),
    )
    sufficient_evidence: bool = Field(
        ...,
        description=(
            "False when the documents do not answer the question, or when the "
            "answer carried no citation that could be checked."
        ),
    )
    generation_mode: str = Field(
        ...,
        description=(
            "llm — written by the configured language model. "
            "extractive — assembled verbatim from the source clauses because "
            "no model is configured; never presented as model-written. "
            "none — no answer was generated."
        ),
    )
    generation_provider: Optional[str] = None
    generation_model: Optional[str] = None
    retrieval_strategy: str
    retrieval_components: List[str] = Field(default_factory=list)
    citations: List[CitationOut] = Field(default_factory=list)
    passages: List[PassageOut] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    took_ms: float
    diagnostics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sufficiency, context budget and citation-check detail.",
    )


class AskStatusResponse(BaseModel):
    """Whether Ask can answer, and what it is configured with."""

    ready: bool
    corpus: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    generation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider, model and whether a key is present — never the key.",
    )
    build_ms: Optional[float] = None
    error: Optional[str] = None
