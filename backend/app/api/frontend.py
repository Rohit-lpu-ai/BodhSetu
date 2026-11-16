"""Serve the BodhSetu frontend from the same application as the API.

Why the API serves the pages at all, rather than a second static server on a
second port (which is what the two throwaway preview tools at the repository
root do -- deliberately not named here, because a test asserts that nothing
under ``backend/app`` mentions them, and that guard is worth more than the
convenience of writing their directory names down):

* **One process, one port, one command.** The brief asks for a single
  coherent application the team can start in one click. Two servers means two
  windows, two failure modes and a port-mismatch bug waiting to happen.
* **Same-origin kills CORS.** With the pages and the API on one origin, no
  preflight, no allow-list and no credentials question arises. The CORS
  middleware stays for the preview tools, which still run on their own ports
  and are still useful.
* **No proxy to maintain.** Those preview servers each carry ~130 lines of
  hand-rolled forwarding. Mounting static files deletes that problem instead
  of solving it twice more.

What this module deliberately does *not* do: it does not take over ``/api``,
``/docs``, ``/redoc`` or ``/openapi.json``. The mount is added **last**, so
every route the routers registered wins over a file of the same name, and a
request for an unknown ``/api/...`` path still gets the API's 404 envelope
rather than an HTML page.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.app.db.config import REPO_ROOT

__all__ = ["FRONTEND_DIR", "frontend_available", "mount_frontend"]

LOGGER = logging.getLogger(__name__)

#: The integrated frontend. ``NiyamSetu-frontend-pages/`` next to it is the
#: teammate's original delivery, kept untouched as the design reference; this
#: directory is the copy the application actually serves and the only one that
#: carries the pages added during integration.
FRONTEND_DIR = REPO_ROOT / "frontend"

#: The page a bare ``/`` lands on. index.html is the teammate's landing page.
ENTRY_PAGE = "index.html"


class RevalidatedStaticFiles(StaticFiles):
    """StaticFiles that makes the browser revalidate every file.

    Without this the browser applies a heuristic freshness lifetime to a
    response that carries no ``Cache-Control``, and serves an edited page from
    cache on reload. That is not a theoretical risk: a browser pass during
    integration spent several minutes debugging a fix that was already on
    disk and simply was not being fetched.

    ``no-cache`` rather than ``no-store``: the file may still be cached, it
    just has to be revalidated, so an unchanged page still answers 304 and
    costs nothing. The two preview servers use ``no-store`` for the same
    reason; this is the cheaper form of the same guarantee.
    """

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def frontend_available(directory: Optional[Path] = None) -> bool:
    """True when there is a frontend to serve.

    Checked rather than assumed: the API is useful on its own, and a missing
    ``frontend/`` must degrade to an API-only server, not to a crash at
    import time.
    """
    root = Path(directory) if directory is not None else FRONTEND_DIR
    return root.is_dir() and (root / ENTRY_PAGE).is_file()


def mount_frontend(application: FastAPI, directory: Optional[Path] = None) -> bool:
    """Mount the frontend at ``/``. Returns whether anything was mounted.

    ``html=True`` makes StaticFiles serve ``index.html`` for a directory
    request, which is what lets ``/`` work. The explicit ``/`` route is still
    here because ``html=True`` alone would also serve ``index.html`` for a
    *missing* path, and a typo'd page silently rendering the landing page is
    worse than a 404.
    """
    root = Path(directory) if directory is not None else FRONTEND_DIR
    if not frontend_available(root):
        LOGGER.warning(
            "no frontend at %s; serving the API only (/api, /docs)", root
        )
        return False

    @application.get("/", include_in_schema=False)
    def _index() -> FileResponse:
        return FileResponse(root / ENTRY_PAGE, headers={"Cache-Control": "no-cache"})

    @application.get("/favicon.ico", include_in_schema=False)
    def _favicon() -> FileResponse:
        """Answer the browser's automatic /favicon.ico with the real mark.

        Every page load otherwise logs a 404 for a file nobody asked for, and
        a 404 in the network panel that is *expected* trains you to ignore the
        ones that are not. The file is a PNG served at the .ico path, which
        browsers accept; ``backend/scripts/make_favicon.py`` regenerates it.
        """
        return FileResponse(root / "favicon.png", media_type="image/png")

    @application.get("/app", include_in_schema=False)
    def _app_shortcut() -> RedirectResponse:
        """A memorable alias for the signed-in home, not a second landing page."""
        return RedirectResponse(url="/user_home.html", status_code=307)

    application.mount(
        "/",
        RevalidatedStaticFiles(directory=str(root), html=True),
        name="frontend",
    )
    LOGGER.info("frontend mounted from %s", root)
    return True
