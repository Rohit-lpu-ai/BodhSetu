"""The Ask service: owns the index, its lifetime, and the pipeline.

One index, rebuilt when the corpus changes
------------------------------------------
Indexing 363 clauses with BM25 takes milliseconds; embedding them with a real
model takes about 50 seconds (the benchmark's own ``index_latency_ms``). Doing
either per request is out of the question, so the index is built once and held.

It is invalidated by a *fingerprint* rather than a timer: ``corpus_counts()``
is one cheap SQL round trip, and ingesting a document moves it. So a document
added through *Add Document* becomes answerable on the next question, without a
restart and without a background job to get wrong. A TTL would either rebuild
constantly or serve a stale corpus after an upload; neither is acceptable for a
feature whose whole claim is that it answers from what is actually stored.

Dense retrieval is optional, loaded in the background, and absent is not broken
------------------------------------------------------------------------------
``sentence-transformers`` may not be installed, the weights may not be
downloaded, and the machine may be offline. None of that should take Ask down —
BM25 alone is a working assistant. So the embedder is loaded on a best-effort
basis, the failure is recorded, and ``active_components`` says ``["bm25"]``
instead of the response pretending to be hybrid.

It is also loaded **off the request path**. Downloading the model and embedding
the corpus took 105 seconds on this machine the first time; doing that inside
the first ``POST /api/ask`` would hand the user a browser timeout instead of an
answer. So ``ensure_ready()`` builds the BM25 index synchronously — milliseconds
— and answers from it immediately, while a daemon thread loads the embedder and
swaps in a hybrid retriever when it is ready. The upgrade is atomic under the
same lock, so a request sees either the lexical retriever or the hybrid one,
never a half-built index. ``GET /api/ask/status`` reports ``dense_state`` as
``loading``, ``ready``, ``unavailable`` or ``disabled``, so the UI can say which
is in play rather than inferring it from latency.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.app.ask.corpus import CorpusSnapshot, corpus_fingerprint, load_corpus_from_repository
from backend.app.db.config import load_dotenv
from rag_module.citation.verifier import CitationVerifier
from rag_module.config.settings import Settings as RagSettings
from rag_module.core.models import RAGQuery, RAGResponse
from rag_module.generation.generator import ExtractiveGenerator, GroundedGenerator
from rag_module.generation.providers import (
    BaseLLMProvider,
    LLMError,
    ProviderUnavailable,
    describe_configuration,
    resolve_provider,
)
from rag_module.pipeline import AskPipeline
from rag_module.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger("niyamsetu.ask.service")

__all__ = ["AskService", "AskServiceState", "get_ask_service", "reset_ask_service"]


@dataclass
class AskServiceState:
    """What the service currently holds, for the status endpoint."""

    ready: bool
    corpus: Dict[str, Any]
    retrieval: Dict[str, Any]
    generation: Dict[str, Any]
    build_ms: Optional[float] = None
    error: Optional[str] = None


class AskService:
    """Builds and holds the Ask index; answers one question at a time."""

    def __init__(
        self,
        repository: Any,
        *,
        settings: Optional[RagSettings] = None,
        provider: Optional[BaseLLMProvider] = None,
        embedder: Optional[Any] = None,
        auto_embedder: bool = True,
        user_contact: Optional[str] = None,
    ) -> None:
        self.repository = repository
        self.user_contact = user_contact.strip().lower() if user_contact else None
        # Load `.env` before reading the environment. `rag_module` deliberately
        # imports nothing from `backend`, so it reads `os.environ` and nothing
        # else -- which means somebody has to put `.env` there first. Relying
        # on the backend's own Settings having been constructed earlier works
        # only by call order, and failed exactly that way: a key sitting in
        # `.env` resolved to "no provider configured" because nothing had read
        # the file yet. Loading it here makes the service correct on its own.
        if settings is None:
            load_dotenv()
        self.settings = settings or RagSettings.from_env()
        self._explicit_provider = provider
        self._explicit_embedder = embedder
        self._auto_embedder = auto_embedder

        self._lock = threading.Lock()
        self._retriever: Optional[HybridRetriever] = None
        self._pipeline: Optional[AskPipeline] = None
        self._snapshot: Optional[CorpusSnapshot] = None
        self._build_ms: Optional[float] = None
        self._embedder_error: Optional[str] = None
        self._provider_error: Optional[str] = None
        self._provider: Optional[BaseLLMProvider] = None
        self._generation_mode = "none"
        #: disabled | loading | ready | unavailable
        self._dense_state = "disabled"
        self._dense_thread: Optional[threading.Thread] = None
        self._dense_ms: Optional[float] = None

    # ------------------------------------------------------------- building

    def _load_embedder(self) -> Optional[Any]:
        """Best effort. A missing model degrades hybrid to BM25, not to an error."""
        if self._explicit_embedder is not None:
            return self._explicit_embedder
        if not self._auto_embedder or not getattr(self.settings, "enable_dense", True):
            return None
        try:
            from rag_module.embeddings.sentence_transformer import (
                SentenceTransformerEmbedder,
            )

            embedder = SentenceTransformerEmbedder(
                model_name_or_path=self.settings.embedding_model_name,
                device=self.settings.embedding_device,
                batch_size=self.settings.embedding_batch_size,
            )
            # Force the lazy load now, so a missing package or missing weights
            # is a build-time fact rather than a surprise inside a request.
            embedder.embed_query("warmup")
            return embedder
        except Exception as exc:  # noqa: BLE001 - every failure means "no dense arm"
            self._embedder_error = f"{type(exc).__name__}: {exc}"
            logger.info(
                "ask: dense retrieval unavailable (%s); running BM25 only",
                type(exc).__name__,
            )
            return None

    def _build_generator(self):
        """The configured model, or the labelled extractive fallback."""
        provider = self._explicit_provider
        if provider is None:
            try:
                provider = resolve_provider(self.settings)
            except LLMError as exc:
                self._provider_error = str(exc)
                provider = None

        if provider is None:
            self._provider = None
            self._generation_mode = "extractive"
            reason = self._provider_error or (
                "No language model is configured (NIYAMSETU_RAG_LLM_PROVIDER is "
                "unset), so this answer is assembled from the source text "
                "rather than written."
            )
            return ExtractiveGenerator(
                reason=reason,
                context_char_budget=self.settings.context_char_budget,
                max_passage_chars=self.settings.max_passage_chars,
            )

        self._provider = provider
        self._generation_mode = "llm"
        return GroundedGenerator(
            provider,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            timeout_seconds=float(
                getattr(self.settings, "llm_timeout_seconds", 60.0)
            ),
            context_char_budget=self.settings.context_char_budget,
            max_passage_chars=self.settings.max_passage_chars,
        )

    def _install(self, snapshot: CorpusSnapshot, retriever: HybridRetriever) -> None:
        """Swap in a retriever and the pipeline over it. The caller holds the lock."""
        self._retriever = retriever
        self._snapshot = snapshot
        self._pipeline = AskPipeline(
            retriever,
            generator=self._build_generator(),
            verifier=CitationVerifier(),
            document_titles=snapshot.document_titles,
            top_k=max(self.settings.retrieval_top_k, 6),
            candidate_depth=self.settings.candidate_depth,
            context_char_budget=self.settings.context_char_budget,
            max_passage_chars=self.settings.max_passage_chars,
        )

    def ensure_ready(self, *, force: bool = False) -> None:
        """Build the lexical index now; upgrade to hybrid in the background."""
        with self._lock:
            current = corpus_fingerprint(self.repository)
            fresh = (
                self._pipeline is not None
                and self._snapshot is not None
                and self._snapshot.fingerprint == current
                and current != "unknown"
            )
            if fresh and not force:
                return

            started = time.perf_counter()
            snapshot = load_corpus_from_repository(self.repository)

            # BM25 only, and synchronously: this is milliseconds, so the very
            # first question is answerable straight away.
            retriever = HybridRetriever(embedder=None, rrf_k=self.settings.rrf_k)
            if snapshot.segments:
                retriever.index(snapshot.segments)
            self._install(snapshot, retriever)
            self._build_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "ask: lexical index built - %d segments, generation=%s, %.0f ms",
                snapshot.size,
                self._generation_mode,
                self._build_ms,
            )

            want_dense = self._explicit_embedder is not None or (
                self._auto_embedder and getattr(self.settings, "enable_dense", True)
            )
            if not want_dense:
                self._dense_state = "disabled"
                self._embedder_error = (
                    "dense retrieval is switched off (NIYAMSETU_RAG_ENABLE_DENSE=0)"
                )
                return
            if not snapshot.segments:
                self._dense_state = "disabled"
                return

            self._dense_state = "loading"

        # Outside the lock: the download and the embedding pass are slow, and
        # holding the lock through them would block every question meanwhile.
        self._start_dense_upgrade(snapshot)

    def _start_dense_upgrade(self, snapshot: CorpusSnapshot) -> None:
        """Load the embedder and swap in a hybrid retriever, off the request path."""
        if self._dense_thread is not None and self._dense_thread.is_alive():
            return

        def _run() -> None:
            started = time.perf_counter()
            embedder = self._load_embedder()
            if embedder is None:
                with self._lock:
                    self._dense_state = "unavailable"
                return
            try:
                retriever = HybridRetriever(
                    embedder=embedder, rrf_k=self.settings.rrf_k
                )
                retriever.index(snapshot.segments)
            except Exception as exc:  # noqa: BLE001 - keep BM25 serving
                self._embedder_error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._dense_state = "unavailable"
                logger.warning("ask: dense index failed (%s)", type(exc).__name__)
                return

            with self._lock:
                # The corpus may have moved while this ran; install only if the
                # snapshot this thread embedded is still the current one.
                still_current = (
                    self._snapshot is not None
                    and self._snapshot.fingerprint == snapshot.fingerprint
                )
                if still_current:
                    self._install(snapshot, retriever)
                    self._dense_state = "ready"
                    self._dense_ms = (time.perf_counter() - started) * 1000.0
                    logger.info(
                        "ask: hybrid index ready - %d segments, %.0f ms",
                        snapshot.size,
                        self._dense_ms,
                    )
                else:
                    logger.info("ask: discarding dense index for a superseded corpus")

        thread = threading.Thread(target=_run, name="ask-dense-index", daemon=True)
        self._dense_thread = thread
        thread.start()

    def wait_for_dense(self, timeout: float = 600.0) -> bool:
        """Block until the dense arm is ready. For tests and warm-up scripts."""
        thread = self._dense_thread
        if thread is None:
            return self._dense_state == "ready"
        thread.join(timeout)
        return self._dense_state == "ready"

    # ---------------------------------------------------------------- state

    def state(self) -> AskServiceState:
        """Everything the status endpoint reports. Never includes a key."""
        try:
            self.ensure_ready()
        except Exception as exc:  # noqa: BLE001
            logger.exception("ask: index build failed")
            return AskServiceState(
                ready=False,
                corpus={},
                retrieval={},
                generation={},
                error=f"{type(exc).__name__}: {exc}",
            )

        snapshot = self._snapshot
        retriever = self._retriever
        generation = describe_configuration(self.settings)
        generation.update(
            {
                "mode": self._generation_mode,
                "active": self._provider is not None,
                "error": self._provider_error,
            }
        )
        return AskServiceState(
            ready=bool(snapshot and snapshot.size > 0),
            corpus=(snapshot.summary() if snapshot else {}),
            retrieval={
                "components": retriever.active_components if retriever else [],
                "is_hybrid": bool(retriever and retriever.is_hybrid),
                "rrf_k": retriever.rrf_k if retriever else None,
                "dense_state": self._dense_state,
                "dense_error": self._embedder_error,
                "dense_ms": self._dense_ms,
                "embedding_model": self.settings.embedding_model_name,
            },
            generation=generation,
            build_ms=self._build_ms,
        )

    # ------------------------------------------------------------------ ask

    def ask(self, query: RAGQuery) -> RAGResponse:
        """Answer one question against the live corpus."""
        self.ensure_ready()
        if self._pipeline is None or not (self._snapshot and self._snapshot.size):
            return self._empty_corpus_response(query)
        response = self._pipeline.ask(query)
        if self._snapshot is not None:
            response.metadata.setdefault("corpus", self._snapshot.summary())
        return response

    def _empty_corpus_response(self, query: RAGQuery) -> RAGResponse:
        """There is nothing to answer from, and saying so is the answer."""
        return RAGResponse(
            query=query.question,
            answer=(
                "There are no documents in the collection yet, so there is "
                "nothing to answer from. Add an official policy PDF through "
                "Add Document and ask again."
            ),
            sufficient_evidence=False,
            citations=[],
            retrieved_passages=[],
            segmentation_method=query.segmentation_method,
            retrieval_strategy="hybrid"
            if (self._retriever and self._retriever.is_hybrid)
            else "bm25",
            latency_ms=0.0,
            metadata={
                "status": "empty_corpus",
                "generation": {"mode": "none"},
                "retrieval": {"components": [], "corpus_segments": 0},
                "corpus": (self._snapshot.summary() if self._snapshot else {}),
                "warnings": [],
            },
        )


# --------------------------------------------------------------------------- #
# Process-wide instance
#
# The index is expensive enough that it must outlive a request, so the service
# is a module-level singleton keyed to nothing: one application, one corpus.
# Tests construct their own AskService directly and never touch this.
# --------------------------------------------------------------------------- #

_SERVICE: Optional[AskService] = None
_SERVICE_LOCK = threading.Lock()


def get_ask_service(repository: Any) -> AskService:
    """The shared service, built on first use."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = AskService(repository)
        return _SERVICE


def reset_ask_service() -> None:
    """Drop the shared service. For tests and for a deliberate reindex."""
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = None
