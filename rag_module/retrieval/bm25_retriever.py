"""BM25 Lexical Retrieval component for NiyamSetu RAG module.

This module implements:
1. BM25Tokenizer: A deterministic regex tokenizer designed for Indian government
   notifications and regulatory documents. It preserves legal references, clause numbers,
   dates, percentages, currency indicators, Indian comma numerals, and Devanagari Unicode
   script without aggressive stop-word removal.
2. compose_searchable_text: A deterministic, non-destructive function that combines
   clause_number, heading, and body text into a weighted composite representation for indexing.
3. BM25Retriever: An in-process lexical retriever compatible with BaseRetriever,
   using a pure-Python, deterministic BM25Okapi implementation with Lucene-style IDF:
   IDF(t) = ln(1.0 + (N - df(t) + 0.5) / (df(t) + 0.5))

Scoring note:
BM25 scores are unbounded ranking scores reflecting term frequency, document frequency,
and length normalization. They are NOT calibrated confidence probabilities.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rag_module.core.interfaces import BaseRetriever
from rag_module.core.models import ClauseSegment, RetrievedPassage

logger = logging.getLogger(__name__)


def compose_searchable_text(segment: ClauseSegment) -> str:
    """Build a composite searchable text representation from a ClauseSegment.

    Combines:
    1. Clause number (with explicit 'clause' prefix and repeat for modest weighting)
    2. Heading (repeated once for modest title weighting)
    3. Segment body text (unmodified)

    This function is purely functional and read-only: it does NOT mutate
    the input ClauseSegment or its text.

    Weighting strategy and document-length trade-off:
    - Repeating clause_number and heading twice provides a modest 2x term-frequency
      (TF) boost for structural metadata, allowing queries like 'clause 3(2)' and
      '3(2)' to match with priority.
    - In single-field BM25, prepending and repeating metadata tokens increases
      the total tokenized document length |D|. As a mathematical consequence of BM25
      document-length normalization (b=0.75), segments with longer metadata face a mild
      score penalty on queries that match only their body text compared to heading-less
      segments.
    """
    parts: List[str] = []

    cn = (segment.clause_number or "").strip()
    if cn:
        parts.append(f"clause {cn} {cn}")

    h = (segment.heading or "").strip()
    if h:
        parts.append(f"{h} {h}")

    parts.append(segment.text or "")
    return " . ".join(parts)


class _PurePythonBM25Okapi:
    """Deterministic, pure-Python BM25 implementation using Lucene-style IDF.

    IDF(t) = ln(1.0 + (N - df(t) + 0.5) / (df(t) + 0.5))

    This formula is strictly non-negative for all valid document frequencies
    (including when a term appears in every document in the corpus), avoiding
    the negative-IDF pathology of Robertson-Spärck Jones un-smoothed BM25.
    """

    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (
            sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0
        )
        self.doc_freqs: List[Dict[str, int]] = []
        self.nd: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]) -> None:
        for doc in corpus:
            frequencies: Dict[str, int] = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                self.nd[word] = self.nd.get(word, 0) + 1

        for word, freq in self.nd.items():
            # Lucene-style BM25 IDF: strictly non-negative for all document frequencies
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: List[str]) -> List[float]:
        """Compute BM25 scores for all documents given a tokenized query."""
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0 or self.avgdl == 0:
            return scores

        for q in query:
            q_idf = self.idf.get(q, 0.0)
            if q_idf <= 0.0:
                continue

            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue

                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / self.avgdl)
                )
                scores[i] += q_idf * (numerator / denominator)

        return scores


class BM25Tokenizer:
    """Deterministic tokenizer designed for Indian government and legal documents.

    Key considerations:
    1. Preserves clause numbers with brackets including mixed forms:
       '(2)', '(iv)', '(2A)', '(1-a)', '(3.1)' via a flexible inner pattern
       that permits hyphens and dots within the brackets.
    2. Preserves complex alphanumeric references: 'RBI/2026-27/300', 'D-SIBs',
       'Section 45-IA', '3.2'.
    3. Preserves percentages ('3.5%') and emits the numeric sub-token ('3.5').
    4. Normalizes currency: '₹' is replaced with 'rs ' before tokenization,
       aligning it with 'Rs.' query terms.
    5. Preserves Indian comma-formatted numerals ('10,000', '5,00,000') and
       emits unformatted versions ('10000', '500000') as sub-tokens.
    6. Preserves Devanagari script words (U+0900-U+0963, U+0966-U+097F).
       The danda (U+0964 '।') and double danda (U+0965 '॥') are excluded from
       the word character class so they do NOT attach to tokens.
    7. Avoids aggressive stop-word removal: legal qualifiers ('shall', 'must',
       'not', 'except', 'prior', 'under') carry binding legal semantics.
    8. Preserves true term frequency: sub-token expansion deduplicates only
       within the current token, never across the entire document.
    """

    # Clause bracket pattern: allows alphanumeric inner content plus '-' and '.'
    # e.g. (2), (iv), (2A), (1-a), (3.1)
    _BRACKET_CLAUSE = r"\([a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*\)"

    TOKEN_PATTERN = re.compile(
        _BRACKET_CLAUSE + r"|"          # clause numbering: (2), (iv), (2A), (1-a), (3.1)
        r"\d+(?:\.\d+)?%|"             # percentages: 3.5%, 100%
        r"\d+(?:,\d+)+|"               # comma-formatted numbers: 10,000, 5,00,000
        r"[a-zA-Z0-9]+(?:[./\-][a-zA-Z0-9]+)+|"  # references: RBI/2026-27/300, D-SIBs
        r"[\w\u0900-\u0963\u0966-\u097F]+",  # alphanumeric + Devanagari (excl. danda U+0964/U+0965)
        re.UNICODE,
    )

    def tokenize(self, text: str) -> List[str]:
        """Convert input text into a list of normalized lexical tokens.

        Steps:
        1. Normalize '₹' to ' rs ' so rupee-symbol queries match 'Rs.' text.
        2. Lowercase and strip.
        3. Apply TOKEN_PATTERN to extract raw tokens.
        4. For each raw token apply sub-token expansion with per-token deduplication
           (preserving true document-level term frequencies):
           - Clause bracket: '(1-a)' -> inner '1-a', then parts '1', 'a'.
           - Percentage: '3.5%' -> numeric '3.5'.
           - Comma number: '5,00,000' -> unformatted '500000'.
           - Complex reference: 'rbi/2026-27/300' -> 'rbi', '2026', '27', '300'.
        """
        if not text:
            return []

        clean_text = text.replace("₹", " rs ").lower().strip()
        raw_tokens = self.TOKEN_PATTERN.findall(clean_text)
        tokens: List[str] = []

        for t in raw_tokens:
            tokens.append(t)
            seen_for_t: Set[str] = {t}

            # Clause bracket: '(2)' -> '2'; '(1-a)' -> '1-a', '1', 'a'; '(3.1)' -> '3.1', '3', '1'
            if t.startswith("(") and t.endswith(")") and len(t) > 2:
                inner = t[1:-1]
                if inner not in seen_for_t:
                    seen_for_t.add(inner)
                    tokens.append(inner)
                # Expand compound inners like '1-a' or '3.1'
                if any(c in inner for c in ("-", ".")):
                    for part in re.findall(r"[a-z0-9]+", inner):
                        if part not in seen_for_t:
                            seen_for_t.add(part)
                            tokens.append(part)

            # Percentage: '3.5%' -> '3.5'
            elif t.endswith("%") and len(t) > 1:
                num = t[:-1]
                if num not in seen_for_t:
                    seen_for_t.add(num)
                    tokens.append(num)

            # Comma-formatted number: '5,00,000' -> '500000'
            elif "," in t and t.replace(",", "").isdigit():
                unformatted = t.replace(",", "")
                if unformatted not in seen_for_t:
                    seen_for_t.add(unformatted)
                    tokens.append(unformatted)

            # Complex alphanumeric reference: 'rbi/2026-27/300' -> 'rbi', '2026', '27', '300'
            elif any(c in t for c in ("/", "-", ".")):
                subparts = re.findall(r"[a-z0-9]+", t)
                for sp in subparts:
                    if sp not in seen_for_t:
                        seen_for_t.add(sp)
                        tokens.append(sp)

        return tokens


class BM25Retriever(BaseRetriever):
    """In-memory BM25 lexical retriever for ClauseSegment passages.

    Compatible with BaseRetriever. Indexes composite representations of
    clause_number, heading, and text while preserving original ClauseSegment
    objects and provenance untouched. Supports filtering by document_id and version_id.
    """

    def __init__(
        self,
        tokenizer: Optional[BM25Tokenizer] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenizer = tokenizer or BM25Tokenizer()
        self.k1 = k1
        self.b = b

        self._segments: List[ClauseSegment] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[_PurePythonBM25Okapi] = None
        self._clause_ids: Set[str] = set()

    def index(self, segments: List[ClauseSegment]) -> None:
        """Build the BM25 index from a list of ClauseSegment objects.

        Replaces any previous index state. Deduplicates segments by clause_id.
        Preserves original ClauseSegment objects and provenance unmodified.
        Searchable text is composed from clause_number, heading, and body text.
        """
        self._segments = []
        self._tokenized_corpus = []
        self._clause_ids = set()
        self._bm25 = None

        if not segments:
            return

        for seg in segments:
            # Deduplicate by clause_id within the index build
            if seg.clause_id in self._clause_ids:
                continue

            self._clause_ids.add(seg.clause_id)
            self._segments.append(seg)
            # Compose searchable representation (clause_number + heading + text)
            searchable_text = compose_searchable_text(seg)
            tokens = self.tokenizer.tokenize(searchable_text)
            self._tokenized_corpus.append(tokens)

        if not self._tokenized_corpus:
            return

        # Instantiate authoritative pure-Python Lucene BM25 engine
        self._bm25 = _PurePythonBM25Okapi(
            self._tokenized_corpus, k1=self.k1, b=self.b
        )

    @property
    def corpus_size(self) -> int:
        """Number of indexed segments."""
        return len(self._segments)

    def term_idf(self, term: str) -> float:
        """Lucene BM25 IDF for one token, including tokens not in the corpus.

        Exposed because the Ask pipeline's sufficiency gate needs to know how
        *distinctive* a question word is before it can judge whether the
        retrieved passages actually cover the question. A term the corpus has
        never seen is the strongest possible signal that the question is about
        something the collection does not contain -- so it is scored at the
        df=0 ceiling rather than being skipped, which is what the retrieval
        path does (an unseen term simply contributes no score there).
        """
        if self._bm25 is None:
            return 0.0
        known = self._bm25.idf.get(term)
        if known is not None:
            return float(known)
        corpus_size = self._bm25.corpus_size
        if corpus_size <= 0:
            return 0.0
        return float(math.log(1.0 + (corpus_size + 0.5) / 0.5))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> List[RetrievedPassage]:
        """Retrieve top-k relevant segments for a query using BM25 scoring.

        Args:
            query: User search query or question.
            top_k: Maximum number of passages to return (clamped >= 1).
            document_id: Optional document filter.
            version_id: Optional version filter.

        Returns:
            List of RetrievedPassage objects ranked by score (descending).
        """
        if not query or not query.strip():
            return []

        if self._bm25 is None or not self._segments:
            return []

        top_k = max(1, top_k)
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)

        # Pair scores with segments and apply filters
        candidates: List[Tuple[float, ClauseSegment]] = []
        for score, seg in zip(raw_scores, self._segments):
            # Only consider positive lexical matches
            if score <= 0.0:
                continue

            if document_id is not None and seg.document_id != document_id:
                continue

            if version_id is not None and seg.version_id != version_id:
                continue

            candidates.append((float(score), seg))

        # Deterministic sorting: score (descending), then ordinal (ascending), then clause_id (ascending)
        candidates.sort(key=lambda x: (-x[0], x[1].ordinal, x[1].clause_id))

        top_candidates = candidates[:top_k]

        results: List[RetrievedPassage] = []
        for rank, (score, seg) in enumerate(top_candidates, start=1):
            results.append(
                RetrievedPassage(
                    segment=seg,
                    score=score,
                    rank=rank,
                    retrieval_method="bm25",
                )
            )

        return results
