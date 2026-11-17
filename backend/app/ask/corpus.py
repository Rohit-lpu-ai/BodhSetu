"""Build the RAG corpus from the database the application is actually serving.

Why not the JSONL export
------------------------
``exports/niyamsetu_segments_501.jsonl`` is a snapshot of the original
four-document development corpus, and the RAG benchmark was measured on it —
so it stays exactly where it is, and ``rag_module/evaluation/`` keeps using it.
It is the wrong thing to *serve from*, for one decisive reason: its ids are
``niyamsetu``'s ids. The running application reads whichever database ``.env``
names, and ``public_id`` comes from a per-database sequence, so ``DOC-0001`` in
the export and ``DOC-0001`` in ``niyamsetu_app`` are different documents. An
answer indexed from the export would cite ``CLS-0000166`` and link the reader
to a clause that says something else entirely — a citation that resolves to the
wrong text is worse than no citation at all.

So the index is built from the live corpus, through the same repository the
rest of the API reads. That also means a document ingested through *Add
Document* is answerable as soon as the index refreshes, with no second
catalogue to keep in step.

Why clause-aware only
---------------------
Both segmentation methods are stored for every version, and they cover the same
words twice. Indexing both would put two views of one provision into retrieval,
where they would occupy two of the top-k slots and read to a language model as
two independent sources corroborating each other. Clause-aware is the one
indexed because it is the finer division, it carries ``clause_number`` and
``heading`` (which the BM25 composite text weights, and which a citation needs
to be readable), and it is the condition the retrieval benchmark measured.
``include_paragraph=True`` is available for experiments and is off by default.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_module.core.models import ClauseSegment, PageProvenance

logger = logging.getLogger("niyamsetu.ask.corpus")

__all__ = ["CorpusSnapshot", "load_corpus_from_repository", "corpus_fingerprint"]

#: The repository caps a page of segments at 100; this walks every page.
_PAGE_SIZE = 100

#: A hard ceiling so a pathological corpus cannot exhaust memory during a
#: request. 50,000 clauses is ~100x the current corpus and still indexes in
#: seconds; crossing it is a signal to move to an on-disk index, not to
#: silently answer from a truncated corpus.
MAX_SEGMENTS = 50_000


@dataclass
class CorpusSnapshot:
    """The segments the index was built from, plus what identifies them."""

    segments: List[ClauseSegment] = field(default_factory=list)
    document_titles: Dict[str, str] = field(default_factory=dict)
    document_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fingerprint: str = ""
    built_at: float = field(default_factory=time.time)
    truncated: bool = False

    @property
    def size(self) -> int:
        return len(self.segments)

    def summary(self) -> Dict[str, Any]:
        return {
            "segments": self.size,
            "documents": len(self.document_titles),
            "fingerprint": self.fingerprint,
            "built_at": self.built_at,
            "truncated": self.truncated,
        }


def corpus_fingerprint(repository: Any) -> str:
    """A cheap value that changes whenever the corpus does.

    Derived from the row counts the repository already exposes — documents,
    versions, pages and segments. Ingesting, or pointing ``.env`` at a
    different database, moves at least one of those, so the index rebuilds.
    Editing a clause's *text* in place without changing any count would not be
    caught, but nothing in this system does that: the loader refuses to
    overwrite a stored version.
    """
    try:
        counts = repository.corpus_counts()
    except Exception as exc:  # noqa: BLE001 - a fingerprint must not 500 a request
        logger.warning("corpus fingerprint unavailable: %s", type(exc).__name__)
        return "unknown"
    payload = "|".join(
        f"{key}={counts.get(key)}"
        for key in sorted(
            ("documents", "versions", "pages", "segments", "clause_aware_segments")
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_corpus_from_repository(
    repository: Any,
    *,
    method: str = "clause_aware",
    include_paragraph: bool = False,
    max_segments: int = MAX_SEGMENTS,
) -> CorpusSnapshot:
    """Read every stored segment of the latest OK version of every document.

    Only the latest extraction-OK version of each document is indexed. Older
    versions are still *searchable* through Library & Search and still
    *comparable* through Compare — but an assistant that answers from three
    issues of one master circular at once will quote a superseded provision
    beside the one that replaced it, with nothing in the ranking to say which
    is current. Answering from the current text is the right default; Compare
    is where version differences belong.
    """
    snapshot = CorpusSnapshot(fingerprint=corpus_fingerprint(repository))

    offset = 0
    documents: List[Dict[str, Any]] = []
    while True:
        page = repository.list_documents(limit=100, offset=offset, sort="public_id")
        items = page.get("items") or []
        documents.extend(items)
        offset += len(items)
        if not items or offset >= int(page.get("total") or 0):
            break

    methods = [method] + (["paragraph"] if include_paragraph and method != "paragraph" else [])

    for document in documents:
        document_id = document["document_id"]
        snapshot.document_titles[document_id] = document.get("title") or document_id
        snapshot.document_meta[document_id] = {
            "title": document.get("title"),
            "department": document.get("department"),
            "document_type": document.get("document_type"),
            "reference_no": document.get("reference_no"),
            "publication_date": document.get("latest_publication_date"),
        }

        if document.get("extraction_status") not in (None, "ok"):
            # A partial extraction can be missing a clause, and a missing
            # clause in an answer is indistinguishable from the policy not
            # saying it. Compare refuses such versions for the same reason.
            logger.info(
                "ask: skipping %s (extraction_status=%s)",
                document_id,
                document.get("extraction_status"),
            )
            continue

        version_id = document.get("latest_version_id")
        for seg_method in methods:
            for record in _iter_segments(
                repository, document_id, version_id, seg_method
            ):
                if len(snapshot.segments) >= max_segments:
                    snapshot.truncated = True
                    logger.warning(
                        "ask: corpus truncated at %d segments", max_segments
                    )
                    return snapshot
                segment = _to_clause_segment(
                    record,
                    document_id,
                    version_id,
                    seg_method,
                    document_title=snapshot.document_titles.get(document_id),
                )
                if segment is not None:
                    snapshot.segments.append(segment)

    return snapshot


def _iter_segments(
    repository: Any,
    document_id: str,
    version_id: Optional[str],
    method: str,
):
    """Walk every page of one version's segments."""
    offset = 0
    while True:
        try:
            payload = repository.get_segments(
                document_id,
                version_public_id=version_id,
                method=method,
                limit=_PAGE_SIZE,
                offset=offset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ask: could not read %s segments for %s: %s",
                method,
                document_id,
                type(exc).__name__,
            )
            return

        items = payload.get("items") or []
        for item in items:
            yield item

        offset += len(items)
        if not items or offset >= int(payload.get("total") or 0):
            return


def _to_clause_segment(
    record: Dict[str, Any],
    document_id: str,
    version_id: Optional[str],
    method: str,
    *,
    document_title: Optional[str] = None,
) -> Optional[ClauseSegment]:
    """Map one repository row to the RAG module's segment model.

    Returns ``None`` rather than raising for a row the model rejects — an
    empty-text segment, say. One unusable row must not make the whole corpus
    unanswerable, and the count difference is visible in the snapshot summary.
    """
    pages = []
    for page in record.get("pages") or []:
        try:
            pages.append(
                PageProvenance(
                    page_number=int(page["page_number"]),
                    char_start=int(page.get("char_start") or 0),
                    char_end=int(page.get("char_end") or 0),
                    line_index_start=int(page.get("line_index_start") or 0),
                    line_index_end=int(page.get("line_index_end") or 0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not pages:
        return None

    try:
        return ClauseSegment(
            document_id=document_id,
            version_id=str(record.get("version_id") or version_id or ""),
            clause_id=str(record["clause_id"]),
            fingerprint=str(record.get("fingerprint") or record["clause_id"]),
            ordinal=int(record.get("ordinal") or 0),
            segment_type=str(record.get("segment_type") or "clause"),
            text=str(record.get("text") or ""),
            page_start=int(record["page_start"]),
            page_end=int(record["page_end"]),
            pages=pages,
            clause_number=record.get("clause_number"),
            clause_number_ambiguous=bool(record.get("clause_number_ambiguous")),
            heading=record.get("heading"),
            segmentation_method=method,
            offset_basis=str(record.get("offset_basis") or "normalized_text"),
            level=int(record.get("level") or 0),
            marker_style=record.get("marker_style"),
            parent_clause_id=record.get("parent_clause_id"),
            # `metadata` is the reserved per-segment object the clause
            # contract already defines for downstream annotations (CLAUDE.md
            # §12.3). The title goes here rather than onto a new field so a
            # citation can be rendered without a second document lookup, and
            # so rag_module needs no knowledge of documents.
            metadata={
                **dict(record.get("metadata") or {}),
                **({"document_title": document_title} if document_title else {}),
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "ask: skipping unusable segment %s: %s",
            record.get("clause_id"),
            exc,
        )
        return None
