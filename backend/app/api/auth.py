"""Authentication and user management endpoints for BodhSetu frontend.

Provides:
  POST /api/check-user  - checks whether an account exists for a given contact
  POST /api/register    - registers a new user profile
  POST /api/login       - verifies credentials and issues a session token
  POST /api/logout      - invalidates the active session token
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from backend.app.db.engine import connection

LOGGER = logging.getLogger(__name__)

auth_router = APIRouter(tags=["auth"])

TABLE_DDL = """
CREATE TABLE IF NOT EXISTS app_users (
    id SERIAL PRIMARY KEY,
    contact VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    state VARCHAR(100),
    district VARCHAR(100),
    residence_area VARCHAR(100),
    age_group VARCHAR(50),
    preferred_language VARCHAR(50),
    highest_education VARCHAR(100),
    education_status VARCHAR(100),
    primary_occupation VARCHAR(100),
    annual_income VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

SESSION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS app_sessions (
    id SERIAL PRIMARY KEY,
    token VARCHAR(255) UNIQUE NOT NULL,
    contact VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() + INTERVAL '30 days'),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
"""

USER_DOCUMENT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS user_documents (
    user_contact VARCHAR(255) NOT NULL,
    document_public_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_contact, document_public_id)
);
"""

# In-memory fallback if database is unavailable
_MEMORY_USERS: Dict[str, Dict[str, Any]] = {}
_MEMORY_SESSIONS: Dict[str, str] = {}


def _ensure_user_table() -> None:
    try:
        with connection() as conn:
            conn.execute(text(TABLE_DDL))
            conn.execute(text(SESSION_TABLE_DDL))
            conn.execute(text(USER_DOCUMENT_TABLE_DDL))
            conn.commit()
    except Exception as exc:
        LOGGER.warning("Could not verify auth tables in PostgreSQL: %s", exc)


def normalize_contact(contact: Optional[str]) -> Optional[str]:
    if contact is None:
        return None
    cleaned = contact.strip().lower()
    if not cleaned:
        return None
    return cleaned


def create_session_token(contact: str) -> str:
    token = secrets.token_urlsafe(32)
    _MEMORY_SESSIONS[token] = normalize_contact(contact) or contact
    try:
        with connection() as conn:
            conn.execute(
                text(
                    "INSERT INTO app_sessions (token, contact) VALUES (:token, :contact) "
                    "ON CONFLICT (token) DO UPDATE SET contact = EXCLUDED.contact, "
                    "expires_at = NOW() + INTERVAL '30 days'"
                ),
                {"token": token, "contact": normalize_contact(contact)},
            )
            conn.commit()
    except Exception:
        pass
    return token


def resolve_session_contact(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token = token.strip()
    if not token:
        return None
    contact = _MEMORY_SESSIONS.get(token)
    if contact:
        return contact
    try:
        with connection() as conn:
            row = conn.execute(
                text(
                    "SELECT contact FROM app_sessions WHERE token = :token AND expires_at > NOW() LIMIT 1"
                ),
                {"token": token},
            ).first()
            if row:
                return normalize_contact(row[0])
    except Exception:
        return None
    return None


def invalidate_session_token(token: Optional[str]) -> bool:
    if not token:
        return False
    key = token.strip()
    if not key:
        return False
    _MEMORY_SESSIONS.pop(key, None)
    try:
        with connection() as conn:
            conn.execute(text("DELETE FROM app_sessions WHERE token = :token"), {"token": key})
            conn.commit()
        return True
    except Exception:
        return False


def get_user_document_ids(user_contact: str, url: Optional[str] = None) -> List[str]:
    contact = normalize_contact(user_contact)
    if not contact:
        return []
    try:
        with connection(url) as conn:
            rows = conn.execute(
                text(
                    "SELECT document_public_id FROM user_documents WHERE LOWER(user_contact) = :contact"
                ),
                {"contact": contact},
            ).fetchall()
            return [str(row[0]) for row in rows]
    except Exception:
        return []


def document_belongs_to_user(document_id: str, user_contact: str, url: Optional[str] = None) -> bool:
    contact = normalize_contact(user_contact)
    if not contact or not document_id:
        return False
    document_public_id = document_id.strip()
    try:
        with connection(url) as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM user_documents WHERE LOWER(user_contact) = :contact "
                    "AND document_public_id = :doc_id LIMIT 1"
                ),
                {"contact": contact, "doc_id": document_public_id},
            ).first()
            return row is not None
    except Exception:
        return False


def auth_token_from_request(request: Request) -> Optional[str]:
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header:
        scheme, _, value = auth_header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    for key in ("x-auth-token", "x-token"):
        value = request.headers.get(key)
        if value and value.strip():
            return value.strip()
    cookie_value = request.cookies.get("bodhsetu_token") or request.cookies.get("auth_token")
    if cookie_value and cookie_value.strip():
        return cookie_value.strip()
    return None


def get_current_user(request: Request) -> Optional["UserProfile"]:
    token = auth_token_from_request(request)
    contact = resolve_session_contact(token)
    if not contact:
        return None
    try:
        with connection() as conn:
            row = conn.execute(
                text(
                    "SELECT full_name, contact, state, district, residence_area, "
                    "preferred_language, primary_occupation, annual_income "
                    "FROM app_users WHERE LOWER(contact) = LOWER(:contact) LIMIT 1"
                ),
                {"contact": contact},
            ).first()
            if row is None:
                return None
            return UserProfile(
                full_name=row[0],
                contact=row[1],
                state=row[2],
                district=row[3],
                residence_area=row[4],
                preferred_language=row[5],
            )
    except Exception:
        return None


def require_current_user(request: Request) -> "UserProfile":
    user = get_current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return user


def _normalize_contact(contact: str) -> List[str]:
    cleaned = contact.strip().lower()
    variants = [cleaned]
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if len(digits) >= 10:
        core = digits[-10:]
        variants.extend([core, f"+91{core}", f"91{core}", f"0{core}"])
    return list(dict.fromkeys(variants))


def _hash_password(raw_or_hash: str) -> str:
    if raw_or_hash.startswith("sha256:"):
        return raw_or_hash
    digest = hashlib.sha256(raw_or_hash.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _verify_password(raw_or_hash: str, stored_hash: str) -> bool:
    if raw_or_hash == stored_hash:
        return True
    return _hash_password(raw_or_hash) == stored_hash


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class CheckUserRequest(BaseModel):
    contact: str


class CheckUserResponse(BaseModel):
    exists: bool
    contact: str


class RegisterRequest(BaseModel):
    contact: str
    password: str
    full_name: str
    state: Optional[str] = None
    district: Optional[str] = None
    residence_area: Optional[str] = None
    age_group: Optional[str] = None
    preferred_language: Optional[str] = None
    highest_education: Optional[str] = None
    education_status: Optional[str] = None
    primary_occupation: Optional[str] = None
    annual_income: Optional[str] = None
    is_active: bool = True


class RegisterResponse(BaseModel):
    success: bool
    token: str
    message: str = "User registered successfully"


class LoginRequest(BaseModel):
    contact: str
    password: str


class UserProfile(BaseModel):
    full_name: str
    contact: str
    state: Optional[str] = None
    district: Optional[str] = None
    residence_area: Optional[str] = None
    preferred_language: Optional[str] = None


class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user: Optional[UserProfile] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@auth_router.post(
    "/check-user",
    response_model=CheckUserResponse,
    summary="Check if a user exists by email or mobile number",
    include_in_schema=False,
)
def check_user(req: CheckUserRequest) -> CheckUserResponse:
    variants = _normalize_contact(req.contact)
    _ensure_user_table()

    # Try PostgreSQL first
    try:
        with connection() as conn:
            stmt = text(
                "SELECT contact FROM app_users WHERE LOWER(contact) = ANY(:variants) LIMIT 1"
            )
            row = conn.execute(stmt, {"variants": variants}).first()
            if row:
                return CheckUserResponse(exists=True, contact=row[0])
    except Exception as exc:
        LOGGER.warning("Database check failed, using fallback: %s", exc)

    # Check memory fallback
    for v in variants:
        if v in _MEMORY_USERS:
            return CheckUserResponse(exists=True, contact=_MEMORY_USERS[v]["contact"])

    return CheckUserResponse(exists=False, contact=req.contact.strip())


@auth_router.post(
    "/register",
    response_model=RegisterResponse,
    summary="Register a new user account",
    include_in_schema=False,
)
def register(req: RegisterRequest) -> RegisterResponse:
    contact_norm = req.contact.strip().lower()
    variants = _normalize_contact(contact_norm)
    pwd_hash = _hash_password(req.password)
    _ensure_user_table()

    # Check existing in PostgreSQL
    try:
        with connection() as conn:
            stmt = text(
                "SELECT id FROM app_users WHERE LOWER(contact) = ANY(:variants) LIMIT 1"
            )
            existing = conn.execute(stmt, {"variants": variants}).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="An account with this email or mobile number already exists.",
                )

            insert_stmt = text(
                """
                INSERT INTO app_users (
                    contact, password_hash, full_name, state, district,
                    residence_area, age_group, preferred_language,
                    highest_education, education_status, primary_occupation,
                    annual_income, is_active
                ) VALUES (
                    :contact, :pwd_hash, :name, :state, :district,
                    :residence, :age_group, :lang,
                    :education, :edu_status, :occupation,
                    :income, :is_active
                )
                """
            )
            conn.execute(
                insert_stmt,
                {
                    "contact": contact_norm,
                    "pwd_hash": pwd_hash,
                    "name": req.full_name.strip(),
                    "state": req.state,
                    "district": req.district,
                    "residence": req.residence_area,
                    "age_group": req.age_group,
                    "lang": req.preferred_language,
                    "education": req.highest_education,
                    "edu_status": req.education_status,
                    "occupation": req.primary_occupation,
                    "income": req.annual_income,
                    "is_active": req.is_active,
                },
            )
            conn.commit()

            token = create_session_token(contact_norm)
            return RegisterResponse(success=True, token=token)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.warning("Database registration failed, saving to fallback: %s", exc)

    # In-memory fallback
    for v in variants:
        if v in _MEMORY_USERS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email or mobile number already exists.",
            )

    token = create_session_token(contact_norm)
    user_record = {
        "contact": contact_norm,
        "password_hash": pwd_hash,
        "full_name": req.full_name.strip(),
        "state": req.state,
        "district": req.district,
        "residence_area": req.residence_area,
        "preferred_language": req.preferred_language,
    }
    _MEMORY_USERS[contact_norm] = user_record
    return RegisterResponse(success=True, token=token)


@auth_router.post(
    "/login",
    response_model=LoginResponse,
    summary="Sign in with email/mobile and password",
    include_in_schema=False,
)
def login(req: LoginRequest) -> LoginResponse:
    contact_norm = req.contact.strip().lower()
    variants = _normalize_contact(contact_norm)
    _ensure_user_table()

    # Try PostgreSQL
    try:
        with connection() as conn:
            stmt = text(
                """
                SELECT contact, password_hash, full_name, state, district,
                       residence_area, preferred_language
                FROM app_users
                WHERE LOWER(contact) = ANY(:variants) AND is_active = TRUE
                LIMIT 1
                """
            )
            row = conn.execute(stmt, {"variants": variants}).first()
            if row:
                stored_hash = row[1]
                if _verify_password(req.password, stored_hash):
                    token = create_session_token(row[0])
                    return LoginResponse(
                        success=True,
                        token=token,
                        user=UserProfile(
                            contact=row[0],
                            full_name=row[2],
                            state=row[3],
                            district=row[4],
                            residence_area=row[5],
                            preferred_language=row[6],
                        ),
                    )
                return LoginResponse(
                    success=False,
                    error="Incorrect password. Please try again.",
                )
    except Exception as exc:
        LOGGER.warning("Database login lookup failed: %s", exc)

    # Check memory fallback
    for v in variants:
        if v in _MEMORY_USERS:
            u = _MEMORY_USERS[v]
            if _verify_password(req.password, u["password_hash"]):
                token = create_session_token(u["contact"])
                return LoginResponse(
                    success=True,
                    token=token,
                    user=UserProfile(
                        contact=u["contact"],
                        full_name=u["full_name"],
                        state=u.get("state"),
                        district=u.get("district"),
                        residence_area=u.get("residence_area"),
                        preferred_language=u.get("preferred_language"),
                    ),
                )
            return LoginResponse(
                success=False,
                error="Incorrect password. Please try again.",
            )

    return LoginResponse(
        success=False,
        error="No account found for this contact. Please sign up first.",
    )


@auth_router.post(
    "/logout",
    response_model=dict,
    summary="Invalidate the active auth token and log the user out",
    include_in_schema=False,
)
def logout(request: Request) -> dict:
    token = auth_token_from_request(request)
    if token:
        invalidate_session_token(token)
    response = JSONResponse({"success": True, "message": "Logged out successfully."})
    response.delete_cookie("bodhsetu_token", path="/")
    response.delete_cookie("auth_token", path="/")
    return response


@auth_router.get(
    "/me",
    response_model=UserProfile,
    summary="Return the authenticated user's profile",
    include_in_schema=False,
)
def me(request: Request) -> UserProfile:
    user = get_current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return user
