"""The Ask endpoints, with no database and no language model.

The service is replaced wholesale through the dependency seam, exactly as the
Compare and Search route tests do — so these run in the suite that needs no
PostgreSQL. The corpus adapter is exercised against a fake repository that
returns the shapes the real one returns, which is what pins the mapping
between the repository's rows and ``rag_module``'s segment model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.api.dependencies import (  # noqa: E402
    Settings,
    get_database_status,
    get_repository,
)
from backend.app.api.main import create_app  # noqa: E402
from backend.app.ask.api import get_ask_service_dependency  # noqa: E402
from backend.app.ask.corpus import (  # noqa: E402
    corpus_fingerprint,
    load_corpus_from_repository,
)
from backend.app.ask.service import AskService  # noqa: E402
from rag_module.generation.providers import MockProvider, ProviderUnavailable  # noqa: E402

CLAUSE_TEXT = (
    "3.2 Banks shall set up special cells at their head offices to monitor the "
    "flow of credit to Scheduled Castes and Scheduled Tribes."
)
SECOND_TEXT = "4.1 Leverage ratio requirements applicable to commercial banks."


class FakeRepository:
    """Returns the shapes ``backend.app.db.repository`` returns."""

    def __init__(self, *, documents: List[Dict[str, Any]] | None = None,
                 segments: Dict[str, List[Dict[str, Any]]] | None = None) -> None:
        self._documents = documents if documents is not None else [
            {
                "document_id": "DOC-0001",
                "title": "Master Circular - Credit facilities to SCs and STs",
                "department": "Reserve Bank of India",
                "document_type": "circular",
                "reference_no": "RBI/2019-20/06",
                "latest_version_id": "VER-0002",
                "latest_publication_date": "2019-07-01",
                "extraction_status": "ok",
                "version_count": 2,
                "page_count": 8,
                "clause_count": 2,
            }
        ]
        self._segments = segments if segments is not None else {
            "DOC-0001": [
                _row("CLS-0000166", CLAUSE_TEXT, ordinal=1, clause_number="3.2"),
                _row("CLS-0000170", SECOND_TEXT, ordinal=2, clause_number="4.1"),
            ]
        }
        self.counts = {
            "documents": 1,
            "versions": 2,
            "pages": 16,
            "segments": 2,
            "clause_aware_segments": 2,
        }

    def corpus_counts(self) -> Dict[str, Any]:
        return dict(self.counts)

    def list_documents(self, **kwargs: Any) -> Dict[str, Any]:
        return {"total": len(self._documents), "items": list(self._documents)}

    def get_segments(self, document_id: str, **kwargs: Any) -> Dict[str, Any]:
        if kwargs.get("method") != "clause_aware":
            return {"total": 0, "items": []}
        items = self._segments.get(document_id, [])
        return {"total": len(items), "items": list(items)}


def _row(clause_id: str, text: str, *, ordinal: int, clause_number: str) -> Dict[str, Any]:
    return {
        "clause_id": clause_id,
        "fingerprint": f"SEG-{clause_id}",
        "ordinal": ordinal,
        "segment_type": "clause",
        "clause_number": clause_number,
        "clause_number_ambiguous": False,
        "heading": None,
        "text": text,
        "page_start": 7,
        "page_end": 7,
        "pages": [
            {
                "page_number": 7,
                "char_start": 100,
                "char_end": 100 + len(text),
                "line_index_start": 0,
                "line_index_end": 2,
            }
        ],
        "level": 1,
        "marker_style": "numeric",
        "parent_clause_id": None,
        "metadata": {},
        "version_id": "VER-0002",
        "offset_basis": "normalized_text",
    }


def make_client(service: AskService) -> TestClient:
    app = create_app(Settings())
    app.dependency_overrides[get_repository] = lambda: object()
    app.dependency_overrides[get_database_status] = lambda: "ok"
    app.dependency_overrides[get_ask_service_dependency] = lambda: service
    return TestClient(app)


def unconfigured_settings():
    """RAG settings with no model, regardless of what `.env` holds.

    These tests assert the *extractive* path, which only runs when no provider
    is configured. They used to build `AskService` with no settings and rely
    on the developer's `.env` being empty — so the moment a real Groq key was
    added they started exercising the LLM path and failed. A unit test must
    not change what it covers based on whether somebody has a key on disk.
    """
    from rag_module.config.settings import Settings as RagSettings

    base = RagSettings()
    return type(base)(
        **{**base.__dict__, "llm_provider": "none", "llm_api_key": None}
    )


@pytest.fixture()
def repository() -> FakeRepository:
    return FakeRepository()


@pytest.fixture()
def service(repository: FakeRepository) -> AskService:
    """A service with no embedder and the scripted provider."""
    return AskService(
        repository,
        provider=MockProvider(reply="Banks must set up special cells [S1]."),
        auto_embedder=False,
    )


@pytest.fixture()
def client(service: AskService) -> TestClient:
    return make_client(service)


# --------------------------------------------------------------------------- #
# The corpus adapter
# --------------------------------------------------------------------------- #


def test_the_corpus_is_built_from_the_repository(repository: FakeRepository) -> None:
    snapshot = load_corpus_from_repository(repository)
    assert snapshot.size == 2
    assert {s.clause_id for s in snapshot.segments} == {"CLS-0000166", "CLS-0000170"}
    assert snapshot.document_titles["DOC-0001"].startswith("Master Circular")


def test_page_provenance_survives_the_mapping(repository: FakeRepository) -> None:
    segment = load_corpus_from_repository(repository).segments[0]
    assert segment.page_start == 7
    assert segment.page_end == 7
    assert segment.pages[0].char_start == 100
    assert segment.offset_basis == "normalized_text"


def test_the_document_title_rides_along_for_citations(repository: FakeRepository) -> None:
    segment = load_corpus_from_repository(repository).segments[0]
    assert segment.metadata["document_title"].startswith("Master Circular")


def test_paragraph_segments_are_not_indexed_by_default(repository: FakeRepository) -> None:
    """Indexing both methods would put two views of one clause into retrieval."""
    assert all(
        s.segmentation_method == "clause_aware"
        for s in load_corpus_from_repository(repository).segments
    )


def test_a_partially_extracted_document_is_skipped() -> None:
    repo = FakeRepository(
        documents=[
            {
                "document_id": "DOC-0001",
                "title": "Partial",
                "latest_version_id": "VER-0001",
                "extraction_status": "partial",
            }
        ]
    )
    assert load_corpus_from_repository(repo).size == 0


def test_an_unusable_row_is_skipped_not_fatal() -> None:
    bad = _row("CLS-BAD", "text", ordinal=1, clause_number="1")
    bad["pages"] = []
    repo = FakeRepository(segments={"DOC-0001": [bad, _row("CLS-OK", CLAUSE_TEXT,
                                                           ordinal=2, clause_number="2")]})
    snapshot = load_corpus_from_repository(repo)
    assert {s.clause_id for s in snapshot.segments} == {"CLS-OK"}


def test_the_fingerprint_moves_when_the_corpus_does(repository: FakeRepository) -> None:
    before = corpus_fingerprint(repository)
    repository.counts["segments"] += 1
    assert corpus_fingerprint(repository) != before


def test_the_fingerprint_is_stable_otherwise(repository: FakeRepository) -> None:
    assert corpus_fingerprint(repository) == corpus_fingerprint(repository)


def test_an_unreadable_repository_does_not_raise() -> None:
    class Broken:
        def corpus_counts(self):
            raise RuntimeError("database down")

    assert corpus_fingerprint(Broken()) == "unknown"


# --------------------------------------------------------------------------- #
# POST /api/ask
# --------------------------------------------------------------------------- #


def test_a_question_is_answered_with_citations(client: TestClient) -> None:
    response = client.post(
        "/api/ask", json={"question": "What special cells must banks set up?"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["sufficient_evidence"] is True
    assert body["generation_mode"] == "llm"
    assert body["citations"]
    citation = body["citations"][0]
    assert citation["clause_id"] == "CLS-0000166"
    assert citation["document_id"] == "DOC-0001"
    assert citation["version_id"] == "VER-0002"
    assert citation["page_start"] == 7
    assert citation["verification_status"] == "STRICT_VERBATIM"
    assert citation["quote"] in CLAUSE_TEXT


def test_the_citation_carries_what_a_deep_link_needs(client: TestClient) -> None:
    """document.html?document_id=…&version_id=…&clause_id=… must be buildable."""
    body = client.post("/api/ask", json={"question": "special cells"}).json()
    citation = body["citations"][0]
    for field in ("document_id", "version_id", "clause_id", "page_start", "page_end"):
        assert citation[field] is not None


def test_retrieved_passages_are_returned_with_fusion_detail(client: TestClient) -> None:
    body = client.post("/api/ask", json={"question": "special cells"}).json()
    assert body["passages"]
    passage = body["passages"][0]
    assert passage["clause_id"]
    assert passage["pages"][0]["char_start"] == 100
    assert "bm25" in passage["component_ranks"]
    assert passage["cited"] is True


def test_passages_can_be_suppressed(client: TestClient) -> None:
    body = client.post(
        "/api/ask", json={"question": "special cells", "include_passages": False}
    ).json()
    assert body["passages"] == []
    assert body["citations"]


def test_an_empty_question_is_a_422_envelope(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_a_missing_question_is_rejected(client: TestClient) -> None:
    assert client.post("/api/ask", json={}).status_code == 422


def test_an_over_long_question_is_reported_not_truncated(client: TestClient) -> None:
    body = client.post("/api/ask", json={"question": "x " * 900}).json()
    assert body["status"] == "question_too_long"
    assert body["sufficient_evidence"] is False


def test_an_off_topic_question_refuses_and_names_the_gap(client: TestClient) -> None:
    body = client.post(
        "/api/ask", json={"question": "How do I renew my passport?"}
    ).json()
    assert body["status"] == "insufficient_evidence"
    assert body["sufficient_evidence"] is False
    assert body["citations"] == []
    assert "passport" in body["answer"]


def test_top_k_is_validated(client: TestClient) -> None:
    assert client.post("/api/ask", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "top_k": 99}).status_code == 422


def test_an_unknown_method_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/ask", json={"question": "x", "method": "not-a-method"}
    )
    assert response.status_code == 422


def test_a_provider_failure_is_a_stated_status(repository: FakeRepository) -> None:
    service = AskService(
        repository,
        provider=MockProvider(fail_with=ProviderUnavailable("no key configured")),
        auto_embedder=False,
    )
    body = make_client(service).post(
        "/api/ask", json={"question": "What special cells must banks set up?"}
    ).json()
    assert body["status"] == "provider_unavailable"
    assert body["sufficient_evidence"] is False
    assert "No key configured" in body["answer"]
    # The evidence still comes back, so the UI can show it.
    assert body["passages"]


def test_an_invented_source_is_stripped_and_warned(repository: FakeRepository) -> None:
    service = AskService(
        repository,
        provider=MockProvider(reply="Real [S1]. Invented [S9]."),
        auto_embedder=False,
    )
    body = make_client(service).post(
        "/api/ask", json={"question": "What special cells must banks set up?"}
    ).json()
    assert "[S9]" not in body["answer"]
    assert "[S1]" in body["answer"]
    assert any("not given" in w for w in body["warnings"])


def test_an_uncited_answer_is_not_called_sufficient(repository: FakeRepository) -> None:
    service = AskService(
        repository,
        provider=MockProvider(reply="A confident claim with no source."),
        auto_embedder=False,
    )
    body = make_client(service).post(
        "/api/ask", json={"question": "What special cells must banks set up?"}
    ).json()
    assert body["sufficient_evidence"] is False
    assert any("caution" in w for w in body["warnings"])


def test_an_empty_corpus_says_so(client: TestClient) -> None:
    empty = AskService(
        FakeRepository(documents=[], segments={}), auto_embedder=False
    )
    body = make_client(empty).post("/api/ask", json={"question": "anything"}).json()
    assert body["status"] == "empty_corpus"
    assert "Add Document" in body["answer"]


def test_an_internal_failure_is_an_error_envelope_not_a_traceback(
    repository: FakeRepository,
) -> None:
    class Exploding(AskService):
        def ask(self, query):  # type: ignore[override]
            raise RuntimeError("internal detail that must not reach the browser")

    response = make_client(Exploding(repository, auto_embedder=False)).post(
        "/api/ask", json={"question": "anything"}
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "internal detail" not in response.text


def test_the_response_schema_has_everything_the_ui_needs(client: TestClient) -> None:
    body = client.post("/api/ask", json={"question": "special cells"}).json()
    for field in (
        "question", "answer", "status", "sufficient_evidence", "generation_mode",
        "retrieval_strategy", "retrieval_components", "citations", "passages",
        "warnings", "took_ms", "diagnostics",
    ):
        assert field in body, field


# --------------------------------------------------------------------------- #
# GET /api/ask/status
# --------------------------------------------------------------------------- #


def test_status_reports_the_corpus_and_components(client: TestClient) -> None:
    body = client.get("/api/ask/status").json()
    assert body["ready"] is True
    assert body["corpus"]["segments"] == 2
    assert body["retrieval"]["components"] == ["bm25"]
    assert body["retrieval"]["is_hybrid"] is False


def test_status_never_contains_the_api_key(repository: FakeRepository) -> None:
    from rag_module.config.settings import Settings as RagSettings

    settings = RagSettings.from_env()
    settings = type(settings)(**{**settings.__dict__, "llm_api_key": "sk-secret-value"})
    service = AskService(repository, settings=settings, auto_embedder=False)
    response = make_client(service).get("/api/ask/status")
    assert "sk-secret-value" not in response.text
    assert response.json()["generation"]["api_key_present"] is True


def test_status_declares_the_extractive_mode_when_no_model_is_configured(
    repository: FakeRepository,
) -> None:
    service = AskService(
        repository, settings=unconfigured_settings(), auto_embedder=False
    )
    body = make_client(service).get("/api/ask/status").json()
    assert body["generation"]["mode"] == "extractive"
    assert body["generation"]["active"] is False
    assert body["generation"]["provider"] == "none"


def test_the_extractive_answer_is_never_labelled_llm(
    repository: FakeRepository,
) -> None:
    service = AskService(
        repository, settings=unconfigured_settings(), auto_embedder=False
    )
    body = make_client(service).post(
        "/api/ask", json={"question": "What special cells must banks set up?"}
    ).json()
    assert body["generation_mode"] == "extractive"
    assert body["generation_provider"] is None
    assert "rather than written" in body["answer"]
    # Still fully cited: the no-model path is not a degraded-evidence path.
    assert body["citations"]


# --------------------------------------------------------------------------- #
# Index lifecycle
# --------------------------------------------------------------------------- #


def test_the_index_rebuilds_when_the_corpus_changes(repository: FakeRepository) -> None:
    service = AskService(repository, auto_embedder=False)
    service.ensure_ready()
    assert service.state().corpus["segments"] == 2

    repository._segments["DOC-0001"].append(
        _row("CLS-0000171", "5.1 A newly ingested clause about banks.",
             ordinal=3, clause_number="5.1")
    )
    repository.counts["segments"] = 3
    repository.counts["clause_aware_segments"] = 3

    assert service.state().corpus["segments"] == 3


def test_the_index_is_not_rebuilt_when_nothing_changed(
    repository: FakeRepository,
) -> None:
    service = AskService(repository, auto_embedder=False)
    service.ensure_ready()
    first = service.state().corpus["built_at"]
    service.ensure_ready()
    assert service.state().corpus["built_at"] == first


def test_the_service_reads_dotenv_so_a_configured_key_is_found(
    repository: FakeRepository, monkeypatch
) -> None:
    """A provider in `.env` must reach the service without a load order.

    `rag_module` reads `os.environ` and nothing else, so something has to put
    `.env` there first. Relying on the backend's own Settings having been
    constructed earlier worked only by accident, and failed exactly that way:
    a real key sitting in `.env` resolved to "no provider configured".
    """
    monkeypatch.delenv("NIYAMSETU_RAG_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("NIYAMSETU_RAG_LLM_API_KEY", raising=False)

    import backend.app.ask.service as service_module

    loaded = {}

    def fake_load_dotenv(*args, **kwargs):
        loaded["called"] = True
        return {}

    monkeypatch.setattr(service_module, "load_dotenv", fake_load_dotenv)
    AskService(repository, auto_embedder=False)
    assert loaded.get("called"), "AskService must load .env before reading the environment"


def test_explicit_settings_skip_dotenv(repository: FakeRepository, monkeypatch) -> None:
    """A caller that supplies Settings owns configuration entirely.

    That is what makes the tests above hermetic: passing settings must not let
    the file on disk back in.
    """
    import backend.app.ask.service as service_module

    loaded = {}
    monkeypatch.setattr(
        service_module, "load_dotenv", lambda *a, **k: loaded.setdefault("called", True)
    )
    AskService(
        repository, settings=unconfigured_settings(), auto_embedder=False
    )
    assert "called" not in loaded
