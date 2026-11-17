"""Core data models for NiyamSetu RAG module.

This module defines the canonical domain models consumed and produced by the RAG
subsystems. The schemas strictly preserve all provenance and structural fields
produced by Person 1's backend pipeline (CLAUDE.md section 12, backend integration
contract v1.0).

All models are dataclasses with strict type hints, validation checks, and clean
to_dict() / from_dict() serialization methods to ensure compatibility without
forcing runtime dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PageProvenance:
    """Provenance for a segment touching a specific physical PDF page.

    The character range [char_start, char_end) indexes into the version's
    normalized_text (the offset basis). Line indexes are 0-based.
    """

    page_number: int
    char_start: int
    char_end: int
    line_index_start: int = 0
    line_index_end: int = 0

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if self.char_start < 0:
            raise ValueError(f"char_start must be >= 0, got {self.char_start}")
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) cannot be less than char_start ({self.char_start})"
            )
        if self.line_index_start < 0 or self.line_index_end < 0:
            raise ValueError("line indexes must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "line_index_start": self.line_index_start,
            "line_index_end": self.line_index_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageProvenance:
        return cls(
            page_number=int(data["page_number"]),
            char_start=int(data["char_start"]),
            char_end=int(data["char_end"]),
            line_index_start=int(data.get("line_index_start", 0)),
            line_index_end=int(data.get("line_index_end", 0)),
        )


@dataclass
class ClauseSegment:
    """A single clause or paragraph segment produced by the ingestion pipeline.

    Preserves the full provenance and hierarchy required for citation
    verification and down-stream academic evaluation.
    """

    document_id: str
    version_id: str
    clause_id: str
    fingerprint: str
    ordinal: int
    segment_type: str
    text: str
    page_start: int
    page_end: int
    pages: List[PageProvenance]
    clause_number: Optional[str] = None
    clause_number_ambiguous: bool = False
    heading: Optional[str] = None
    segmentation_method: str = "clause_aware"
    offset_basis: str = "normalized_text"
    level: int = 0
    marker_style: Optional[str] = None
    parent_clause_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.clause_id:
            raise ValueError("clause_id must not be empty")
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.text:
            raise ValueError("text must not be empty")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError(
                f"invalid page range: page_start={self.page_start}, page_end={self.page_end}"
            )
        if not self.pages:
            raise ValueError("pages list must contain at least one PageProvenance entry")

    @property
    def char_start(self) -> int:
        """First character offset on page_start for single-page segments.

        Raises:
            ValueError: If the segment spans multiple pages. Character offsets
                in NiyamSetu index into each page's normalized_text separately,
                so there is no continuous document-level character offset across pages.
                Use char_span_for_page(page_number) instead.
        """
        if self.page_start != self.page_end:
            raise ValueError(
                f"Cannot access flat char_start on multi-page segment ({self.clause_id} spans "
                f"pages {self.page_start}-{self.page_end}). Use char_span_for_page(page_number) instead."
            )
        return self.pages[0].char_start

    @property
    def char_end(self) -> int:
        """Last character offset on page_end for single-page segments.

        Raises:
            ValueError: If the segment spans multiple pages. Use char_span_for_page(page_number) instead.
        """
        if self.page_start != self.page_end:
            raise ValueError(
                f"Cannot access flat char_end on multi-page segment ({self.clause_id} spans "
                f"pages {self.page_start}-{self.page_end}). Use char_span_for_page(page_number) instead."
            )
        return self.pages[-1].char_end

    def char_span_for_page(self, page_number: int) -> Tuple[int, int]:
        """Return (char_start, char_end) for a specific touched page in normalized_text.

        Args:
            page_number: 1-indexed physical page number.

        Returns:
            Tuple of [char_start, char_end) offsets into that page's normalized_text.

        Raises:
            ValueError: If page_number is not one of the pages touched by this segment.
        """
        for p in self.pages:
            if p.page_number == page_number:
                return (p.char_start, p.char_end)
        touched = [p.page_number for p in self.pages]
        raise ValueError(
            f"Page {page_number} is not part of clause {self.clause_id} (touches pages: {touched})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "clause_id": self.clause_id,
            "fingerprint": self.fingerprint,
            "ordinal": self.ordinal,
            "segment_type": self.segment_type,
            "clause_number": self.clause_number,
            "clause_number_ambiguous": self.clause_number_ambiguous,
            "heading": self.heading,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": [p.to_dict() for p in self.pages],
            "char_start": self.pages[0].char_start if self.page_start == self.page_end else None,
            "char_end": self.pages[0].char_end if self.page_start == self.page_end else None,
            "segmentation_method": self.segmentation_method,
            "offset_basis": self.offset_basis,
            "level": self.level,
            "marker_style": self.marker_style,
            "parent_clause_id": self.parent_clause_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        default_document_id: Optional[str] = None,
        default_version_id: Optional[str] = None,
        default_method: str = "clause_aware",
        default_offset_basis: str = "normalized_text",
    ) -> ClauseSegment:
        pages_raw = data.get("pages") or []
        pages = [PageProvenance.from_dict(p) for p in pages_raw]

        return cls(
            document_id=data.get("document_id") or default_document_id or "",
            version_id=data.get("version_id") or default_version_id or "",
            clause_id=str(data["clause_id"]),
            fingerprint=str(data.get("fingerprint", "")),
            ordinal=int(data.get("ordinal", 0)),
            segment_type=str(data.get("segment_type", "clause")),
            clause_number=data.get("clause_number"),
            clause_number_ambiguous=bool(data.get("clause_number_ambiguous", False)),
            heading=data.get("heading"),
            text=str(data["text"]),
            page_start=int(data.get("page_start", pages[0].page_number if pages else 1)),
            page_end=int(data.get("page_end", pages[-1].page_number if pages else 1)),
            pages=pages,
            segmentation_method=str(data.get("segmentation_method", default_method)),
            offset_basis=str(data.get("offset_basis", default_offset_basis)),
            level=int(data.get("level", 0)),
            marker_style=data.get("marker_style"),
            parent_clause_id=data.get("parent_clause_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ClauseListResponse:
    """Envelope returned by GET /api/documents/{id}/clauses."""

    document_id: str
    version_id: str
    segmentation_method: str
    offset_basis: str
    total: int
    limit: int
    offset: int
    items: List[ClauseSegment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "segmentation_method": self.segmentation_method,
            "offset_basis": self.offset_basis,
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClauseListResponse:
        doc_id = data.get("document_id", "")
        ver_id = data.get("version_id", "")
        method = data.get("segmentation_method", "clause_aware")
        basis = data.get("offset_basis", "normalized_text")
        raw_items = data.get("items") or []

        items = [
            ClauseSegment.from_dict(
                item,
                default_document_id=doc_id,
                default_version_id=ver_id,
                default_method=method,
                default_offset_basis=basis,
            )
            for item in raw_items
        ]
        return cls(
            document_id=doc_id,
            version_id=ver_id,
            segmentation_method=method,
            offset_basis=basis,
            total=int(data.get("total", len(items))),
            limit=int(data.get("limit", len(items))),
            offset=int(data.get("offset", 0)),
            items=items,
        )


@dataclass
class DocumentSummary:
    """Brief metadata summary for a document in the NiyamSetu catalog."""

    document_id: str
    title: str
    department: Optional[str] = None
    document_type: Optional[str] = None
    reference_no: Optional[str] = None
    version_count: int = 1
    latest_publication_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentSummary:
        return cls(
            document_id=str(data["document_id"]),
            title=str(data["title"]),
            department=data.get("department"),
            document_type=data.get("document_type"),
            reference_no=data.get("reference_no"),
            version_count=int(data.get("version_count", 1)),
            latest_publication_date=data.get("latest_publication_date"),
        )


@dataclass
class RAGQuery:
    """Input query submitted to the RAG module."""

    question: str
    document_ids: Optional[List[str]] = None
    version_ids: Optional[List[str]] = None
    segmentation_method: str = "clause_aware"
    retrieval_strategy: str = "hybrid"
    top_k: int = 5
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("question must not be empty")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.segmentation_method not in {"clause_aware", "paragraph"}:
            raise ValueError(
                f"segmentation_method must be 'clause_aware' or 'paragraph', got {self.segmentation_method}"
            )
        if self.retrieval_strategy not in {"bm25", "dense", "hybrid"}:
            raise ValueError(
                f"retrieval_strategy must be 'bm25', 'dense', or 'hybrid', got {self.retrieval_strategy}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RAGQuery:
        return cls(
            question=str(data["question"]),
            document_ids=data.get("document_ids"),
            version_ids=data.get("version_ids"),
            segmentation_method=str(data.get("segmentation_method", "clause_aware")),
            retrieval_strategy=str(data.get("retrieval_strategy", "hybrid")),
            top_k=int(data.get("top_k", 5)),
            metadata_filters=dict(data.get("metadata_filters") or {}),
        )


@dataclass
class RetrievedPassage:
    """A scored passage returned by a retriever."""

    segment: ClauseSegment
    score: float
    rank: int
    retrieval_method: str

    @property
    def clause_id(self) -> str:
        return self.segment.clause_id

    @property
    def document_id(self) -> str:
        return self.segment.document_id

    @property
    def version_id(self) -> str:
        return self.segment.version_id

    @property
    def text(self) -> str:
        return self.segment.text

    @property
    def page_start(self) -> int:
        return self.segment.page_start

    @property
    def page_end(self) -> int:
        return self.segment.page_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "score": self.score,
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "segment": self.segment.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetrievedPassage:
        segment = ClauseSegment.from_dict(data["segment"])
        return cls(
            segment=segment,
            score=float(data["score"]),
            rank=int(data["rank"]),
            retrieval_method=str(data["retrieval_method"]),
        )


@dataclass
class Citation:
    """A source citation backing an answer claim.

    Preserves full traceability to document, version, clause, and page range.
    Supports honest verification accounting for extraction anomalies (e.g. broken words).
    """

    document_id: str
    version_id: str
    clause_id: str
    page_start: int
    page_end: int
    quote: str
    page_number: Optional[int] = None
    verification_status: str = "UNVERIFIED"
    extracted_raw_quote: Optional[str] = None
    char_offset: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        valid_statuses = {
            "UNVERIFIED",
            "STRICT_VERBATIM",
            "AUDITED_SPAN_MATCH",
            "REJECTED",
        }
        if self.verification_status not in valid_statuses:
            raise ValueError(
                f"invalid verification_status: {self.verification_status}. Must be one of {valid_statuses}"
            )
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError(
                f"invalid page range: page_start={self.page_start}, page_end={self.page_end}"
            )

        if self.char_offset is not None:
            if self.page_number is None:
                raise ValueError(
                    "page_number must be provided when char_offset is specified "
                    "(offsets are per-page into normalized_text)"
                )
            start, end = self.char_offset
            if start < 0:
                raise ValueError(f"char_offset start ({start}) must be >= 0")
            if end < start:
                raise ValueError(
                    f"char_offset end ({end}) cannot be less than start ({start})"
                )

        if self.page_number is not None:
            if self.page_number < self.page_start or self.page_number > self.page_end:
                raise ValueError(
                    f"page_number ({self.page_number}) must be between page_start ({self.page_start}) "
                    f"and page_end ({self.page_end})"
                )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "clause_id": self.clause_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "quote": self.quote,
            "page_number": self.page_number,
            "verification_status": self.verification_status,
        }
        if self.extracted_raw_quote is not None:
            res["extracted_raw_quote"] = self.extracted_raw_quote
        if self.char_offset is not None:
            res["char_offset"] = list(self.char_offset)
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Citation:
        char_offset = None
        if "char_offset" in data and data["char_offset"] is not None:
            raw_offset = data["char_offset"]
            char_offset = (int(raw_offset[0]), int(raw_offset[1]))
        page_number = data.get("page_number")
        if page_number is not None:
            page_number = int(page_number)
        return cls(
            document_id=str(data["document_id"]),
            version_id=str(data["version_id"]),
            clause_id=str(data["clause_id"]),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            quote=str(data["quote"]),
            page_number=page_number,
            verification_status=str(data.get("verification_status", "UNVERIFIED")),
            extracted_raw_quote=data.get("extracted_raw_quote"),
            char_offset=char_offset,
        )


@dataclass
class RAGResponse:
    """Final output emitted by the RAG pipeline."""

    query: str
    answer: str
    sufficient_evidence: bool
    citations: List[Citation] = field(default_factory=list)
    retrieved_passages: List[RetrievedPassage] = field(default_factory=list)
    segmentation_method: str = "clause_aware"
    retrieval_strategy: str = "hybrid"
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sufficient_evidence": self.sufficient_evidence,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_passages": [p.to_dict() for p in self.retrieved_passages],
            "segmentation_method": self.segmentation_method,
            "retrieval_strategy": self.retrieval_strategy,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RAGResponse:
        citations = [Citation.from_dict(c) for c in data.get("citations", [])]
        passages = [
            RetrievedPassage.from_dict(p) for p in data.get("retrieved_passages", [])
        ]
        return cls(
            query=str(data["query"]),
            answer=str(data["answer"]),
            sufficient_evidence=bool(data["sufficient_evidence"]),
            citations=citations,
            retrieved_passages=passages,
            segmentation_method=str(data.get("segmentation_method", "clause_aware")),
            retrieval_strategy=str(data.get("retrieval_strategy", "hybrid")),
            latency_ms=data.get("latency_ms"),
            metadata=dict(data.get("metadata") or {}),
        )

