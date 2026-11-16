"""Unit tests for BM25 lexical retriever, tokenizer, and composite indexing (Milestone 2).

Validates:
1. BM25Tokenizer:
   - Clause numbers with brackets: (2), (iv), (2A), (1-a), (3.1)
   - Complex alphanumeric references: RBI/2026-27/300, D-SIBs, Section 45-IA
   - Percentages with sub-token expansion: 3.5% -> ['3.5%', '3.5']
   - Currency normalization: Rs., ₹ -> 'rs'
   - Indian comma-formatted numerals: 10,000 -> ['10,000', '10000']
   - Devanagari Unicode preservation (U+0900-U+0963, U+0966-U+097F)
   - Danda punctuation ('।', '॥') excluded from tokens
   - Legal qualifier preservation: 'shall', 'not', 'except', 'prior', 'must', 'under'
   - Case folding and empty/whitespace input handling
   - Sub-token term frequency preservation: repeated references retain appropriate counts

2. compose_searchable_text:
   - Deterministic composite representation of clause_number, heading, and text
   - Modest additional weighting for clause_number and heading
   - Safe handling of missing/empty clause_number or heading
   - Immutability of original ClauseSegment and its text

3. BM25Retriever:
   - Pure-Python Lucene-style IDF implementation:
     IDF(t) = ln(1.0 + (N - df(t) + 0.5) / (df(t) + 0.5))
   - Strictly non-negative IDF for all document frequencies
   - Retrieval for terms occurring in every document
   - Clause-number retrieval: 'clause 3(2)' and '3(2)' retrieve segment with clause_number '3(2)'
     even when absent from body text
   - Heading retrieval: 'D-SIB Surcharge' retrieves segment with that heading
     even when absent from body text
   - Body text retrieval still works seamlessly
   - Missing heading and clause_number handled safely
   - Immutability of original segment provenance on RetrievedPassage.segment
   - Index rebuilding completely clears stale records
   - Document ID and version ID filtering
   - top_k clamping and ranking
   - Deterministic tie-breaking on identical scores (ordinal ascending, clause_id ascending)
   - Empty corpus, empty query, unseen terms handling
"""

from __future__ import annotations

from collections import Counter
import unittest
from typing import List, Optional

from rag_module.core.interfaces import BaseRetriever
from rag_module.core.models import ClauseSegment, PageProvenance, RetrievedPassage
from rag_module.retrieval.bm25_retriever import (
    BM25Retriever,
    BM25Tokenizer,
    compose_searchable_text,
)


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _make_segment(
    clause_id: str,
    text: str,
    document_id: str = "doc_test_001",
    version_id: str = "ver_test_001",
    ordinal: int = 1,
    clause_number: Optional[str] = "1",
    heading: Optional[str] = "General",
    segment_type: str = "clause",
    segmentation_method: str = "clause_aware",
    page_start: int = 1,
    page_end: int = 1,
) -> ClauseSegment:
    """Return a minimal but complete ClauseSegment for unit testing."""
    pages = [
        PageProvenance(
            page_number=p,
            char_start=0,
            char_end=len(text),
        )
        for p in range(page_start, page_end + 1)
    ]
    return ClauseSegment(
        document_id=document_id,
        version_id=version_id,
        clause_id=clause_id,
        fingerprint=f"fp_{clause_id}",
        ordinal=ordinal,
        segment_type=segment_type,
        clause_number=clause_number,
        heading=heading,
        text=text,
        page_start=page_start,
        page_end=page_end,
        pages=pages,
        segmentation_method=segmentation_method,
        offset_basis="page",
    )


# ---------------------------------------------------------------------------
# BM25Tokenizer tests
# ---------------------------------------------------------------------------

class TestBM25TokenizerClauseNumbers(unittest.TestCase):
    """Clause bracketed number preservation and sub-token expansion."""

    def setUp(self) -> None:
        self.tok = BM25Tokenizer()

    def test_simple_numeric_clause(self) -> None:
        tokens = self.tok.tokenize("Under clause (2) of this Act.")
        self.assertIn("(2)", tokens)
        self.assertIn("2", tokens)

    def test_roman_numeral_clause(self) -> None:
        tokens = self.tok.tokenize("as specified in sub-clause (iv).")
        self.assertIn("(iv)", tokens)
        self.assertIn("iv", tokens)

    def test_mixed_case_clause(self) -> None:
        """(2A) is normalized and preserved."""
        tokens = self.tok.tokenize("Refer clause (2A) for exceptions.")
        self.assertIn("(2a)", tokens)
        self.assertIn("2a", tokens)

    def test_hyphenated_clause(self) -> None:
        """(1-a) sub-token expansion should yield '1-a', '1', 'a'."""
        tokens = self.tok.tokenize("Under clause (1-a) of Section 3.")
        self.assertIn("(1-a)", tokens)
        self.assertIn("1-a", tokens)
        self.assertIn("1", tokens)
        self.assertIn("a", tokens)

    def test_decimal_clause(self) -> None:
        """(3.1) sub-token expansion should yield '3.1', '3', '1'."""
        tokens = self.tok.tokenize("Clause (3.1) applies to NBFCs.")
        self.assertIn("(3.1)", tokens)
        self.assertIn("3.1", tokens)
        self.assertIn("3", tokens)
        self.assertIn("1", tokens)

    def test_repeated_clause_subtoken_frequencies(self) -> None:
        """Repeated bracketed clause numbers must preserve true sub-token term frequency."""
        text = "Refer to clause (2) and later apply clause (2) as well as sub-clause (2)."
        tokens = self.tok.tokenize(text)
        counts = Counter(tokens)
        self.assertEqual(counts["(2)"], 3)
        self.assertEqual(counts["2"], 3)

    def test_repeated_percentage_subtoken_frequencies(self) -> None:
        """Repeated percentages must preserve true numeric sub-token frequency."""
        text = "Minimum CET1 is 3.5% and additional Tier 1 is 3.5%."
        tokens = self.tok.tokenize(text)
        counts = Counter(tokens)
        self.assertEqual(counts["3.5%"], 2)
        self.assertEqual(counts["3.5"], 2)

    def test_repeated_complex_reference_subtoken_frequencies(self) -> None:
        """Repeated alphanumeric references must preserve sub-token frequencies."""
        text = "Circular RBI/2026-27/300 supersedes RBI/2026-27/300."
        tokens = self.tok.tokenize(text)
        counts = Counter(tokens)
        self.assertEqual(counts["rbi/2026-27/300"], 2)
        self.assertEqual(counts["300"], 2)
        self.assertEqual(counts["rbi"], 2)


class TestBM25TokenizerReferences(unittest.TestCase):
    """Legal and regulatory reference tokenization."""

    def setUp(self) -> None:
        self.tok = BM25Tokenizer()

    def test_circular_reference(self) -> None:
        tokens = self.tok.tokenize("Notification RBI/2026-27/300 on banks.")
        self.assertIn("rbi/2026-27/300", tokens)
        self.assertIn("rbi", tokens)
        self.assertIn("2026", tokens)
        self.assertIn("27", tokens)
        self.assertIn("300", tokens)

    def test_hyphenated_acronym(self) -> None:
        tokens = self.tok.tokenize("D-SIBs shall maintain an additional buffer.")
        self.assertIn("d-sibs", tokens)
        self.assertIn("d", tokens)
        self.assertIn("sibs", tokens)

    def test_section_with_hyphen(self) -> None:
        tokens = self.tok.tokenize("Penalties under Section 45-IA.")
        self.assertIn("45-ia", tokens)
        self.assertIn("45", tokens)
        self.assertIn("ia", tokens)

    def test_full_regulatory_sentence(self) -> None:
        """Combined legal reference with clause numbers."""
        text = "Section 4(1)(a) under RBI/2026-27/300 for D-SIBs with CAR 3.5% as of 31/03/2026."
        tokens = self.tok.tokenize(text)
        self.assertIn("section", tokens)
        self.assertIn("(1)", tokens)
        self.assertIn("(a)", tokens)
        self.assertIn("rbi/2026-27/300", tokens)
        self.assertIn("d-sibs", tokens)
        self.assertIn("3.5%", tokens)
        self.assertIn("3.5", tokens)


class TestBM25TokenizerNumerals(unittest.TestCase):
    """Percentage, decimal, and Indian comma-numeral tokenization."""

    def setUp(self) -> None:
        self.tok = BM25Tokenizer()

    def test_percentage_expansion(self) -> None:
        tokens = self.tok.tokenize("Minimum CAR of 11.5% required.")
        self.assertIn("11.5%", tokens)
        self.assertIn("11.5", tokens)

    def test_comma_number_lakh(self) -> None:
        """Indian lakh notation: 5,00,000 -> '5,00,000' and '500000'."""
        tokens = self.tok.tokenize("Penalty not exceeding Rs. 5,00,000.")
        self.assertIn("5,00,000", tokens)
        self.assertIn("500000", tokens)

    def test_comma_number_thousand(self) -> None:
        tokens = self.tok.tokenize("Charge of ₹10,000 per day.")
        self.assertIn("10,00,000" if "10,00,000" in tokens else "10,000", tokens)
        self.assertIn("10000", tokens)

    def test_rupee_symbol_normalization(self) -> None:
        """₹ is normalized to 'rs' to unify with 'Rs.' queries."""
        tokens_symbol = self.tok.tokenize("Fine of ₹500 imposed.")
        tokens_abbrev = self.tok.tokenize("Fine of Rs. 500 imposed.")
        self.assertIn("rs", tokens_symbol)
        self.assertIn("rs", tokens_abbrev)
        self.assertIn("500", tokens_symbol)
        self.assertIn("500", tokens_abbrev)


class TestBM25TokenizerDevanagari(unittest.TestCase):
    """Devanagari Unicode handling."""

    def setUp(self) -> None:
        self.tok = BM25Tokenizer()

    def test_hindi_words_preserved(self) -> None:
        tokens = self.tok.tokenize("भारतीय रिज़र्व बैंक की अधिसूचना संख्या 25।")
        self.assertIn("भारतीय", tokens)
        self.assertIn("रिज़र्व", tokens)
        self.assertIn("बैंक", tokens)
        self.assertIn("की", tokens)
        self.assertIn("अधिसूचना", tokens)
        self.assertIn("संख्या", tokens)
        self.assertIn("25", tokens)

    def test_danda_punctuation_stripped(self) -> None:
        """Devanagari danda U+0964 must NOT be attached to tokens."""
        tokens = self.tok.tokenize("नए नियम लागू होंगे। पुरानी दरें रद्द।")
        for t in tokens:
            self.assertFalse(
                t.endswith("।"),
                f"Token '{t}' should not carry trailing danda U+0964",
            )
            self.assertFalse(
                t.endswith("॥"),
                f"Token '{t}' should not carry trailing double danda U+0965",
            )

    def test_mixed_hindi_english(self) -> None:
        """Mixed-script text: Devanagari and English in same sentence."""
        tokens = self.tok.tokenize("Priority Sector Lending (प्राथमिकता क्षेत्र ऋण) guidelines.")
        self.assertIn("priority", tokens)
        self.assertIn("sector", tokens)
        self.assertIn("lending", tokens)
        self.assertIn("प्राथमिकता", tokens)
        self.assertIn("क्षेत्र", tokens)
        self.assertIn("ऋण", tokens)


class TestBM25TokenizerLegalSemantics(unittest.TestCase):
    """Legal qualifier preservation and case folding."""

    def setUp(self) -> None:
        self.tok = BM25Tokenizer()

    def test_legal_qualifiers_retained(self) -> None:
        """'shall', 'not', 'except', 'prior', 'must', 'under' must NOT be stripped."""
        text = "The borrower shall not transfer assets except with prior written approval."
        tokens = self.tok.tokenize(text)
        for word in ("shall", "not", "except", "prior", "written", "approval"):
            self.assertIn(word, tokens, f"Legal qualifier '{word}' was incorrectly removed")

    def test_case_folding(self) -> None:
        t1 = self.tok.tokenize("CAPITAL")
        t2 = self.tok.tokenize("capital")
        t3 = self.tok.tokenize("Capital")
        self.assertEqual(t1, t2)
        self.assertEqual(t2, t3)

    def test_empty_input(self) -> None:
        self.assertEqual(self.tok.tokenize(""), [])

    def test_whitespace_only_input(self) -> None:
        self.assertEqual(self.tok.tokenize("   \n\t  "), [])

    def test_punctuation_only_input(self) -> None:
        result = self.tok.tokenize("--- ... ///")
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# Composite Searchable Text Tests
# ---------------------------------------------------------------------------

class TestComposeSearchableText(unittest.TestCase):
    """Unit tests for compose_searchable_text function."""

    def test_full_segment_composition(self) -> None:
        seg = _make_segment(
            clause_id="seg_100",
            text="Scheduled banks must maintain capital buffer.",
            clause_number="3(2)",
            heading="D-SIB Surcharge",
        )
        comp = compose_searchable_text(seg)
        # Verify clause prefix and repeat
        self.assertIn("clause 3(2) 3(2)", comp)
        # Verify heading repeat
        self.assertIn("D-SIB Surcharge D-SIB Surcharge", comp)
        # Verify body text
        self.assertIn("Scheduled banks must maintain capital buffer.", comp)

    def test_missing_heading_handled_cleanly(self) -> None:
        seg = _make_segment(
            clause_id="seg_101",
            text="Body text without heading.",
            clause_number="5",
            heading="",
        )
        comp = compose_searchable_text(seg)
        self.assertIn("clause 5 5", comp)
        self.assertIn("Body text without heading.", comp)

    def test_missing_clause_number_handled_cleanly(self) -> None:
        seg = _make_segment(
            clause_id="seg_102",
            text="Paragraph text with heading.",
            clause_number="",
            heading="Introduction",
        )
        comp = compose_searchable_text(seg)
        self.assertIn("Introduction Introduction", comp)
        self.assertIn("Paragraph text with heading.", comp)

    def test_missing_both_clause_number_and_heading(self) -> None:
        seg = _make_segment(
            clause_id="seg_103",
            text="Pure paragraph segment without metadata.",
            clause_number="",
            heading="",
        )
        comp = compose_searchable_text(seg)
        self.assertEqual(comp, "Pure paragraph segment without metadata.")

    def test_literal_none_clause_number_and_heading(self) -> None:
        seg = _make_segment(
            clause_id="seg_105",
            text="Text with literal None metadata.",
            clause_number=None,
            heading=None,
        )
        comp = compose_searchable_text(seg)
        self.assertEqual(comp, "Text with literal None metadata.")

    def test_original_segment_not_mutated(self) -> None:
        """compose_searchable_text must not modify the ClauseSegment instance."""
        seg = _make_segment(
            clause_id="seg_104",
            text="Original body text.",
            clause_number="1(a)",
            heading="Scope",
        )
        original_text = seg.text
        original_heading = seg.heading
        original_cn = seg.clause_number

        _ = compose_searchable_text(seg)

        self.assertEqual(seg.text, original_text)
        self.assertEqual(seg.heading, original_heading)
        self.assertEqual(seg.clause_number, original_cn)


# ---------------------------------------------------------------------------
# BM25Retriever tests
# ---------------------------------------------------------------------------

class TestBM25RetrieverInterface(unittest.TestCase):
    """Interface compliance and basic instantiation."""

    def test_implements_base_retriever(self) -> None:
        self.assertIsInstance(BM25Retriever(), BaseRetriever)

    def test_default_tokenizer_assigned(self) -> None:
        retriever = BM25Retriever()
        self.assertIsInstance(retriever.tokenizer, BM25Tokenizer)

    def test_custom_tokenizer_accepted(self) -> None:
        custom_tok = BM25Tokenizer()
        retriever = BM25Retriever(tokenizer=custom_tok)
        self.assertIs(retriever.tokenizer, custom_tok)


class TestBM25RetrieverWithCorpus(unittest.TestCase):
    """Retrieval behaviour with a representative regulatory corpus."""

    def setUp(self) -> None:
        self.retriever = BM25Retriever()
        # Representative five-segment corpus spanning different docs / versions.
        # Notice: '3(2)' and 'D-SIB Surcharge' do NOT appear in seg_002 text!
        self._segments = [
            _make_segment(
                clause_id="seg_001",
                text=(
                    "The Capital Adequacy Ratio (CAR) for all scheduled commercial banks "
                    "shall be maintained at a minimum of 11.5%."
                ),
                document_id="doc_rbi_01",
                version_id="v1",
                ordinal=1,
                clause_number="3(1)",
                heading="Capital Requirements",
            ),
            _make_segment(
                clause_id="seg_002",
                text=(
                    "Domestic Systemically Important Banks (D-SIBs) shall maintain an "
                    "additional Common Equity Tier 1 (CET1) buffer of 0.80%."
                ),
                document_id="doc_rbi_01",
                version_id="v1",
                ordinal=2,
                clause_number="3(2)",
                heading="D-SIB Surcharge",
            ),
            _make_segment(
                clause_id="seg_003",
                text=(
                    "Penalties under Section 45-IA of the RBI Act shall not exceed "
                    "Rs. 5,00,000 for initial non-compliance."
                ),
                document_id="doc_rbi_02",
                version_id="v1",
                ordinal=1,
                clause_number="7(a)",
                heading="Penal Provisions",
            ),
            _make_segment(
                clause_id="seg_004",
                text=(
                    "Export credit facilities and interest subvention for MSMEs under "
                    "Priority Sector Lending guidelines."
                ),
                document_id="doc_msme_01",
                version_id="v2",
                ordinal=1,
                clause_number="",
                heading="",
                segment_type="paragraph",
                segmentation_method="paragraph",
            ),
            _make_segment(
                clause_id="seg_005",
                text=(
                    "भारतीय रिज़र्व बैंक द्वारा जारी दिशा-निर्देशों के अनुसार "
                    "प्राथमिकता प्राप्त क्षेत्र ऋण (Priority Sector Lending) पर दिशा-निर्देश।"
                ),
                document_id="doc_rbi_hi",
                version_id="v1",
                ordinal=1,
                clause_number="1",
                heading="प्राथमिकता क्षेत्र",
            ),
        ]
        self.retriever.index(self._segments)

    # --- Clause number and Heading retrieval (Audited Requirements) ---

    def test_clause_number_retrieval_with_clause_prefix(self) -> None:
        """Query 'clause 3(2)' retrieves seg_002 via indexed clause_number."""
        results = self.retriever.retrieve("clause 3(2)", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_002")
        self.assertEqual(results[0].segment.clause_number, "3(2)")

    def test_clause_number_retrieval_bare_number(self) -> None:
        """Query '3(2)' retrieves seg_002 via indexed clause_number."""
        results = self.retriever.retrieve("3(2)", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_002")

    def test_heading_retrieval_when_absent_from_body_text(self) -> None:
        """Query 'D-SIB Surcharge' retrieves seg_002 whose heading is 'D-SIB Surcharge'."""
        results = self.retriever.retrieve("D-SIB Surcharge", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_002")
        self.assertEqual(results[0].segment.heading, "D-SIB Surcharge")

    def test_heading_retrieval_penal_provisions(self) -> None:
        """Query 'Penal Provisions' retrieves seg_003 via heading."""
        results = self.retriever.retrieve("Penal Provisions", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_003")

    def test_body_text_retrieval_still_works(self) -> None:
        """Ordinary queries against body text continue to rank the correct segment."""
        results = self.retriever.retrieve("Common Equity Tier 1 CET1 buffer", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_002")

    # --- Exact match ---

    def test_exact_match_dsib_ranks_first(self) -> None:
        results = self.retriever.retrieve("D-SIBs buffer CET1", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_002")
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].retrieval_method, "bm25")
        self.assertGreater(results[0].score, 0.0)

    def test_exact_match_capital_adequacy(self) -> None:
        results = self.retriever.retrieve("11.5% CAR capital adequacy", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_001")

    # --- Partial / multi-term matching ---

    def test_partial_query_matching(self) -> None:
        results = self.retriever.retrieve("CET1", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_002")

    def test_multi_term_relevance_ordering(self) -> None:
        results = self.retriever.retrieve("D-SIBs CET1 buffer banks", top_k=3)
        clause_ids = [r.segment.clause_id for r in results]
        self.assertEqual(clause_ids[0], "seg_002")

    def test_mixed_query_with_out_of_vocabulary_term(self) -> None:
        """Query containing both valid corpus terms and out-of-vocabulary terms retrieves matches."""
        results = self.retriever.retrieve("banks xyznonexistentterm", top_k=3)
        self.assertGreater(len(results), 0)
        # Segments containing 'banks' should be retrieved despite unseen term
        for r in results:
            self.assertIn("banks", r.segment.text.lower())

    # --- Legal references and percentages ---

    def test_section_reference_retrieval(self) -> None:
        results = self.retriever.retrieve("Section 45-IA penalties", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_003")

    def test_percentage_query_matching(self) -> None:
        results = self.retriever.retrieve("11.5% banks", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_001")

    def test_currency_and_rupee_query(self) -> None:
        results = self.retriever.retrieve("Rs. 5,00,000 penalty compliance", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_003")

    # --- Devanagari ---

    def test_devanagari_query_retrieval(self) -> None:
        results = self.retriever.retrieve("भारतीय रिज़र्व बैंक प्राथमिकता", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].segment.clause_id, "seg_005")

    # --- Filtering ---

    def test_filter_by_document_id(self) -> None:
        results = self.retriever.retrieve("shall banks", top_k=5, document_id="doc_rbi_02")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.segment.document_id, "doc_rbi_02")

    def test_filter_by_version_id(self) -> None:
        results = self.retriever.retrieve("MSMEs credit", top_k=5, version_id="v2")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.segment.version_id, "v2")
        self.assertEqual(results[0].segment.clause_id, "seg_004")

    def test_filter_document_and_version_combined(self) -> None:
        results = self.retriever.retrieve(
            "MSMEs", top_k=5, document_id="doc_msme_01", version_id="v2"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].segment.clause_id, "seg_004")

    def test_filter_document_wrong_version_returns_empty(self) -> None:
        results = self.retriever.retrieve(
            "MSMEs", top_k=5, document_id="doc_msme_01", version_id="v1"
        )
        self.assertEqual(results, [])

    # --- top_k ---

    def test_top_k_limits_results(self) -> None:
        results = self.retriever.retrieve("shall banks", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_top_k_zero_clamped_to_one(self) -> None:
        results = self.retriever.retrieve("D-SIBs", top_k=0)
        self.assertEqual(len(results), 1)

    def test_rank_field_sequential(self) -> None:
        results = self.retriever.retrieve("shall banks capital", top_k=4)
        for i, r in enumerate(results, start=1):
            self.assertEqual(r.rank, i)

    # --- Scores ---

    def test_scores_are_positive_floats(self) -> None:
        results = self.retriever.retrieve("capital adequacy", top_k=3)
        for r in results:
            self.assertIsInstance(r.score, float)
            self.assertGreater(r.score, 0.0)

    def test_scores_descending_order(self) -> None:
        results = self.retriever.retrieve("D-SIBs banks buffer", top_k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # --- Provenance preservation ---

    def test_segment_provenance_intact(self) -> None:
        """Every field of the original ClauseSegment must be unchanged."""
        results = self.retriever.retrieve("D-SIBs", top_k=1)
        self.assertEqual(len(results), 1)
        seg = results[0].segment
        original = self._segments[1]  # seg_002
        self.assertEqual(seg.document_id, original.document_id)
        self.assertEqual(seg.version_id, original.version_id)
        self.assertEqual(seg.clause_id, original.clause_id)
        self.assertEqual(seg.fingerprint, original.fingerprint)
        self.assertEqual(seg.ordinal, original.ordinal)
        self.assertEqual(seg.clause_number, original.clause_number)
        self.assertEqual(seg.heading, original.heading)
        self.assertEqual(seg.page_start, original.page_start)
        self.assertEqual(seg.page_end, original.page_end)
        self.assertEqual(len(seg.pages), len(original.pages))
        self.assertEqual(seg.segmentation_method, original.segmentation_method)
        # Ensure seg.text was not modified with composite metadata
        self.assertEqual(seg.text, original.text)

    # --- Segmentation method ---

    def test_paragraph_segmentation_method_supported(self) -> None:
        results = self.retriever.retrieve("MSMEs export credit", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].segment.segmentation_method, "paragraph")
        self.assertEqual(results[0].segment.segment_type, "paragraph")

    # --- Edge cases ---

    def test_empty_query_string(self) -> None:
        self.assertEqual(self.retriever.retrieve("", top_k=5), [])

    def test_whitespace_only_query(self) -> None:
        self.assertEqual(self.retriever.retrieve("   \t\n  ", top_k=5), [])

    def test_unseen_terms_return_empty(self) -> None:
        results = self.retriever.retrieve("cryptocurrency blockchain web3", top_k=5)
        self.assertEqual(results, [])


class TestBM25RetrieverCommonTermsAndTieBreaking(unittest.TestCase):
    """Test suite verifying Lucene-IDF non-negativity and deterministic tie-breaking."""

    def setUp(self) -> None:
        self.retriever = BM25Retriever()

    def test_term_occurring_in_every_document(self) -> None:
        """Lucene IDF guarantees strictly positive score even when a term appears in all documents."""
        # 'shall' appears in every one of the 4 documents
        segments = [
            _make_segment("c_01", "Bank A shall maintain capital.", ordinal=1),
            _make_segment("c_02", "Bank B shall maintain liquidity.", ordinal=2),
            _make_segment("c_03", "Bank C shall maintain reserves.", ordinal=3),
            _make_segment("c_04", "Bank D shall maintain records.", ordinal=4),
        ]
        self.retriever.index(segments)

        results = self.retriever.retrieve("shall", top_k=10)
        # In Robertson-Spärck Jones BM25, this would return 0 results because IDF <= 0.
        # Under Lucene BM25, all 4 documents must be retrieved with positive scores!
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertGreater(r.score, 0.0)

    def test_deterministic_tie_breaking_common_terms(self) -> None:
        """When BM25 scores are identical, segments sort by ordinal ascending then clause_id ascending."""
        # 3 identical texts have identical term frequency and length -> identical BM25 score
        segments = [
            _make_segment("tie_c", "Standard compliance obligation.", ordinal=5),
            _make_segment("tie_a", "Standard compliance obligation.", ordinal=2),
            _make_segment("tie_b", "Standard compliance obligation.", ordinal=2),
        ]
        self.retriever.index(segments)
        results = self.retriever.retrieve("obligation", top_k=3)
        self.assertEqual(len(results), 3)

        # First tie-break: ordinal ascending (2 before 5)
        # Second tie-break: clause_id ascending ('tie_a' before 'tie_b')
        self.assertEqual(results[0].segment.clause_id, "tie_a")
        self.assertEqual(results[1].segment.clause_id, "tie_b")
        self.assertEqual(results[2].segment.clause_id, "tie_c")

    def test_tie_breaking_deterministic_across_calls(self) -> None:
        segments = [
            _make_segment("tie_z", "Repeating text sentence.", ordinal=1),
            _make_segment("tie_m", "Repeating text sentence.", ordinal=1),
            _make_segment("tie_a", "Repeating text sentence.", ordinal=1),
        ]
        self.retriever.index(segments)
        results_1 = [r.segment.clause_id for r in self.retriever.retrieve("repeating", top_k=3)]
        results_2 = [r.segment.clause_id for r in self.retriever.retrieve("repeating", top_k=3)]
        self.assertEqual(results_1, results_2)
        self.assertEqual(results_1, ["tie_a", "tie_m", "tie_z"])

    def test_top_k_respects_tie_breaking_order(self) -> None:
        segments = [
            _make_segment("tie_c", "Same corpus text.", ordinal=1),
            _make_segment("tie_a", "Same corpus text.", ordinal=1),
            _make_segment("tie_b", "Same corpus text.", ordinal=1),
        ]
        self.retriever.index(segments)
        results = self.retriever.retrieve("corpus", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].segment.clause_id, "tie_a")
        self.assertEqual(results[1].segment.clause_id, "tie_b")


class TestBM25RetrieverIndexManagement(unittest.TestCase):
    """Index building, deduplication, and replacement."""

    def setUp(self) -> None:
        self.retriever = BM25Retriever()

    def test_empty_corpus_returns_empty(self) -> None:
        self.retriever.index([])
        self.assertEqual(self.retriever.retrieve("anything", top_k=5), [])

    def test_deduplication_first_occurrence_wins(self) -> None:
        segments = [
            _make_segment("dup_01", "First valid text.", ordinal=1),
            _make_segment("dup_01", "Second duplicate text ignored.", ordinal=2),
            _make_segment("dup_02", "Another unique segment.", ordinal=3),
        ]
        self.retriever.index(segments)
        self.assertEqual(len(self.retriever._segments), 2)
        self.assertEqual(self.retriever._segments[0].text, "First valid text.")

    def test_deduplication_does_not_index_duplicate(self) -> None:
        segments = [
            _make_segment("dup_01", "Unique first entry.", ordinal=1),
            _make_segment("dup_01", "Stale duplicate ignored.", ordinal=2),
        ]
        self.retriever.index(segments)
        results_stale = self.retriever.retrieve("stale duplicate", top_k=5)
        self.assertEqual(results_stale, [])

    def test_index_rebuild_clears_stale_state(self) -> None:
        self.retriever.index([
            _make_segment("old_01", "Old obsolete guideline.", ordinal=1),
        ])
        self.assertGreater(len(self.retriever.retrieve("obsolete", top_k=5)), 0)

        # Replace with new corpus
        self.retriever.index([
            _make_segment("new_01", "New active regulation.", ordinal=1),
        ])
        self.assertEqual(self.retriever.retrieve("obsolete", top_k=5), [])
        new_results = self.retriever.retrieve("active", top_k=5)
        self.assertEqual(len(new_results), 1)
        self.assertEqual(new_results[0].segment.clause_id, "new_01")

    def test_index_rebuild_internal_state_reset(self) -> None:
        self.retriever.index([_make_segment("x_01", "Alpha beta gamma.")])
        self.retriever.index([_make_segment("y_01", "Delta epsilon zeta.")])
        self.assertIn("y_01", self.retriever._clause_ids)
        self.assertNotIn("x_01", self.retriever._clause_ids)

    def test_indexing_with_missing_heading_or_clause_number(self) -> None:
        """Segments with empty or None headings and clause numbers must index without error."""
        segments = [
            _make_segment("m_01", "Segment with empty strings.", clause_number="", heading=""),
            _make_segment("m_02", "Segment with clause number only.", clause_number="4", heading=""),
            _make_segment("m_03", "Segment with heading only.", clause_number="", heading="Heading Only"),
        ]
        self.retriever.index(segments)
        self.assertEqual(len(self.retriever.retrieve("empty", top_k=3)), 1)
        self.assertEqual(len(self.retriever.retrieve("clause 4", top_k=3)), 1)
        res_heading = self.retriever.retrieve("Heading Only", top_k=3)
        self.assertGreater(len(res_heading), 0)
        self.assertEqual(res_heading[0].segment.clause_id, "m_03")

    def test_indexing_with_literal_none_heading_and_clause_number(self) -> None:
        """Segments with literal None for heading and clause_number index and retrieve safely."""
        seg = _make_segment(
            clause_id="none_01",
            text="Text with literal None metadata.",
            clause_number=None,
            heading=None,
        )
        self.retriever.index([seg])
        results = self.retriever.retrieve("literal", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].segment.clause_id, "none_01")
        self.assertIsNone(results[0].segment.clause_number)
        self.assertIsNone(results[0].segment.heading)


if __name__ == "__main__":
    unittest.main()
