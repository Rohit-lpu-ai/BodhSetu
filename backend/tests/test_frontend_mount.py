"""The frontend is served by the API application, and must not shadow it.

These tests exist because mounting static files at ``/`` is the one change in
this integration that can break the API by accident. A mount at the root
matches every path that was not already routed, so the ordering in
``create_app`` is load-bearing: register it before the routers and
``/api/documents`` becomes a 404 HTML page.

They also pin the two behaviours a browser pass found the hard way:

* every file carries ``Cache-Control``, because a page edited on disk and
  served from the browser's heuristic cache looks exactly like a fix that did
  not work;
* an unknown ``/api/...`` path still answers with the §12.1 error envelope,
  not with the landing page.

No database is needed: the repository and the database check are both
overridden, so these run in the 900-odd tests that need no PostgreSQL.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from backend.app.api.frontend import (  # noqa: E402
    ENTRY_PAGE,
    FRONTEND_DIR,
    frontend_available,
    mount_frontend,
)
from backend.app.api.main import create_app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    """An application with the frontend mounted and no database."""
    app = create_app(Settings())
    app.dependency_overrides[get_repository] = lambda: object()
    app.dependency_overrides[get_database_status] = lambda: "ok"
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def test_the_repository_actually_ships_a_frontend() -> None:
    """If this fails the application is API-only, which is a real regression."""
    assert frontend_available(), f"no {ENTRY_PAGE} under {FRONTEND_DIR}"


def test_every_page_the_navigation_links_to_exists() -> None:
    """A nav entry pointing at a missing file is a 404 in the product."""
    expected = {
        "index.html",
        "login.html",
        "register.html",
        "complete-profile.html",
        "user_home.html",
        "chat.html",
        "explore.html",
        "document.html",
        "updates.html",
        "saved.html",
        "add-document.html",
        "assets/bodhsetu.css",
        "assets/api.js",
        "favicon.png",
    }
    missing = {name for name in expected if not (FRONTEND_DIR / name).is_file()}
    assert not missing, f"missing frontend files: {sorted(missing)}"


def test_a_missing_frontend_directory_degrades_to_api_only(tmp_path) -> None:
    """A checkout without frontend/ must still serve the API, not crash."""
    assert frontend_available(tmp_path) is False

    app = create_app(Settings(serve_frontend=False))
    assert mount_frontend(app, tmp_path) is False

    app.dependency_overrides[get_database_status] = lambda: "ok"
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404


# --------------------------------------------------------------------------- #
# The mount must not shadow the API
# --------------------------------------------------------------------------- #


def test_the_api_still_answers_with_the_frontend_mounted(client: TestClient) -> None:
    assert client.get("/api/health").status_code == 200


def test_an_unknown_api_path_keeps_the_error_envelope(client: TestClient) -> None:
    """Not an HTML page: a consumer parses this body (§12.1)."""
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"]


def test_openapi_and_swagger_survive_the_mount(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_the_mount_adds_no_documented_api_path() -> None:
    """The API contract is unchanged by the mount.

    Sixteen paths since ``backend/app/ask/`` added ``/api/ask`` and
    ``/api/ask/status``; the point of this test is that ``/``, ``/app`` and
    ``/favicon.ico`` add none of their own, because they are all
    ``include_in_schema=False``.
    """
    paths = create_app(Settings()).openapi()["paths"]
    assert len(paths) == 16
    assert all(path.startswith("/api/") for path in paths)
    assert "/" not in paths
    assert "/app" not in paths
    assert "/favicon.ico" not in paths


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #


def test_root_serves_the_landing_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "BodhSetu" in response.text


def test_a_named_page_is_served(client: TestClient) -> None:
    response = client.get("/explore.html")
    assert response.status_code == 200
    assert "Explore" in response.text


def test_a_missing_page_is_404_not_the_landing_page(client: TestClient) -> None:
    """html=True would otherwise answer a typo with index.html."""
    response = client.get("/not-a-page.html")
    assert response.status_code == 404
    assert "From government words" not in response.text


def test_every_served_file_must_be_revalidated(client: TestClient) -> None:
    for path in ("/", "/explore.html", "/assets/api.js", "/assets/bodhsetu.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers.get("Cache-Control") == "no-cache", path


def test_the_favicon_is_served_rather_than_404ing(client: TestClient) -> None:
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_app_redirects_to_the_home_screen(client: TestClient) -> None:
    response = client.get("/app", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/user_home.html"


def test_the_frontend_can_be_switched_off(client: TestClient) -> None:
    app = create_app(Settings(serve_frontend=False))
    app.dependency_overrides[get_database_status] = lambda: "ok"
    api_only = TestClient(app)
    assert api_only.get("/").status_code == 404
    assert api_only.get("/api/health").status_code == 200


# --------------------------------------------------------------------------- #
# What the pages are allowed to contain
# --------------------------------------------------------------------------- #


def test_no_page_calls_a_hardcoded_backend_port() -> None:
    """Same-origin /api only.

    A page that hardcodes 127.0.0.1:8000 works until the app is started on
    another port, and then fails silently with a CORS error rather than a
    visible one. The one permitted mention is the file:// fallback in api.js.
    """
    offenders = []
    for page in sorted(FRONTEND_DIR.glob("*.html")):
        text = page.read_text(encoding="utf-8")
        if "127.0.0.1:8000" in text or "localhost:8000" in text:
            offenders.append(page.name)
    assert not offenders, f"pages hardcoding the API port: {offenders}"


def test_the_pages_added_here_contain_no_invented_policy_content() -> None:
    """Every document string on screen comes from the API, never the markup.

    The brief is explicit that no fake policy examples may remain. These
    markers are the ones a placeholder corpus would have left behind.
    """
    banned = ("Lorem ipsum", "DOC-0042", "Sample Policy", "Example Scheme")
    for page in sorted(FRONTEND_DIR.glob("*.html")):
        text = page.read_text(encoding="utf-8")
        for marker in banned:
            assert marker not in text, f"{page.name} contains placeholder {marker!r}"


def test_ask_is_wired_to_the_real_rag_endpoint() -> None:
    """chat.html calls POST /api/ask, not the clause-search endpoint.

    It used to show keyword hits under a "coming soon" note because no RAG
    existed. Now the pipeline does, and the page must use it -- showing
    search results while implying an assistant answered would be exactly the
    fabrication the earlier note was avoiding.
    """
    text = (FRONTEND_DIR / "chat.html").read_text(encoding="utf-8")
    assert "/api/ask" in text
    assert "/api/ask/status" in text
    assert "Answers are coming soon" not in text


def test_ask_distinguishes_a_generated_answer_from_an_assembled_one() -> None:
    """generation_mode must reach the reader, not just the API.

    An extractive answer is assembled verbatim from the sources because no
    model is configured. Presenting that as model-written prose is the one
    thing this page must never do.
    """
    text = (FRONTEND_DIR / "chat.html").read_text(encoding="utf-8")
    assert "generation_mode" in text
    assert "written by a language model" in text
    assert "no language model configured" in text


def test_ask_renders_document_text_as_nodes_never_as_markup() -> None:
    """The answer and its quotes came out of a PDF by way of a model."""
    text = (FRONTEND_DIR / "chat.html").read_text(encoding="utf-8")
    assert "innerHTML" not in text
    assert "textContent" in text
