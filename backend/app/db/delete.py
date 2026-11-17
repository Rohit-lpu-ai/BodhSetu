"""Document deletion helper: completely removes a document and its hierarchy."""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

from backend.app.db.engine import connection

LOGGER = logging.getLogger(__name__)


def delete_document_by_id(document_identifier: str, url: Optional[str] = None) -> bool:
    """Delete a document and all related versions, segments, and pages."""
    with connection(url) as conn:
        stmt = text(
            "SELECT id, public_id FROM documents "
            "WHERE public_id = :id OR (public_id IS NOT NULL AND id::text = :id) "
            "LIMIT 1"
        )
        row = conn.execute(stmt, {"id": document_identifier}).first()
        if not row:
            return False
        doc_id, public_id = row[0], row[1]

        conn.execute(text("ALTER TABLE segments DISABLE TRIGGER segments_provenance_required"))
        try:
            conn.execute(
                text("""
                    UPDATE segments SET parent_id = NULL
                    WHERE version_id IN (SELECT id FROM document_versions WHERE document_id = :doc_id)
                """),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("""
                    UPDATE document_versions SET supersedes_version_id = NULL
                    WHERE document_id = :doc_id OR supersedes_version_id IN (
                        SELECT id FROM document_versions WHERE document_id = :doc_id
                    )
                """),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("""
                    DELETE FROM segment_pages WHERE segment_id IN (
                        SELECT id FROM segments WHERE version_id IN (
                            SELECT id FROM document_versions WHERE document_id = :doc_id
                        )
                    )
                """),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("""
                    DELETE FROM page_removed_lines WHERE page_id IN (
                        SELECT id FROM pages WHERE version_id IN (
                            SELECT id FROM document_versions WHERE document_id = :doc_id
                        )
                    )
                """),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("DELETE FROM segments WHERE version_id IN (SELECT id FROM document_versions WHERE document_id = :doc_id)"),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("DELETE FROM pages WHERE version_id IN (SELECT id FROM document_versions WHERE document_id = :doc_id)"),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("DELETE FROM document_versions WHERE document_id = :doc_id"),
                {"doc_id": doc_id},
            )
            try:
                conn.execute(
                    text("DELETE FROM user_documents WHERE document_public_id = :pub_id"),
                    {"pub_id": public_id},
                )
            except Exception:
                pass
            conn.execute(
                text("DELETE FROM documents WHERE id = :doc_id"),
                {"doc_id": doc_id},
            )
            conn.commit()
            return True
        finally:
            try:
                conn.execute(text("ALTER TABLE segments ENABLE TRIGGER segments_provenance_required"))
                conn.commit()
            except Exception:
                pass
