import datetime
import hashlib
import hmac
import os
import re
from functools import wraps
from pathlib import Path

import jwt
import requests
from flask import Flask, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

ENV_FILE = Path(__file__).resolve().parent / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def env(key, default=""):
    return str(os.getenv(key, default)).strip().strip("'\"")


SUPABASE_URL = env("SUPABASE_URL").rstrip("/")
SUPABASE_KEY = env("SUPABASE_KEY")
USER_TABLE = env("SUPABASE_USER_TABLE", env("SUPABASE_TABLE", "user_database"))
JWT_SECRET = env("JWT_SECRET_KEY")
JWT_HOURS = int(env("JWT_EXPIRATION_HOURS", "24"))
ALLOWED_ORIGINS = [o.strip() for o in env("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

missing = [name for name, value in (
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_KEY", SUPABASE_KEY),
    ("JWT_SECRET_KEY", JWT_SECRET),
) if not value]
if missing:
    raise RuntimeError("Missing required environment variables: " + ", ".join(missing))

app = Flask(__name__)

EMAIL_PATTERN = re.compile(r"^[^\s@,()&]+@[^\s@,()&]+\.[^\s@,()&]+$")
PROFILE_FIELDS = (
    "full_name", "state", "district", "residence_type", "age_group",
    "preferred_language", "highest_education", "education_status",
    "primary_occupation", "annual_income",
)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if "*" in ALLOWED_ORIGINS:
        allowed = "*"
    elif origin in ALLOWED_ORIGINS:
        allowed = origin
        response.headers["Vary"] = "Origin"
    else:
        allowed = None
    if allowed:
        response.headers["Access-Control-Allow-Origin"] = allowed
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
    return response


def db(method, params=None, body=None):
    try:
        return requests.request(
            method,
            f"{SUPABASE_URL}/rest/v1/{USER_TABLE}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            params=params,
            json=body,
            timeout=15,
        )
    except requests.exceptions.RequestException as exc:
        app.logger.error("Database request failed: %s", type(exc).__name__)
        return None


def error(message, status):
    return jsonify({"success": False, "error": message}), status


def normalize_contact(raw):
    raw = str(raw or "").strip()
    if not raw:
        return None, "", []
    if "@" in raw:
        email = raw.lower()
        if EMAIL_PATTERN.match(email):
            return "email", email, [email]
        return None, "", []
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 10:
        core = digits[-10:]
        return "mobile", core, [core, f"+91{core}", f"91{core}", f"0{core}"]
    return None, "", []


def contact_filter(variants):
    if len(variants) == 1:
        return {"contact": f"eq.{variants[0]}"}
    return {"or": "(" + ",".join(f"contact.eq.{v}" for v in variants) + ")"}


def clean_field(data, key):
    value = data.get(key)
    if value is None:
        return None
    return str(value).strip()[:200] or None


def sanitize_user(user):
    safe = dict(user)
    safe.pop("password", None)
    return safe


def make_token(user):
    user_ui = user.get("user_ui")
    if not user_ui:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "user_ui": user_ui,
        "iat": now,
        "exp": now + datetime.timedelta(hours=JWT_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def token_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return error("Authentication required.", 401)
        try:
            claims = jwt.decode(header.split(" ", 1)[1].strip(), JWT_SECRET, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return error("Invalid or expired session. Please log in again.", 401)
        return view(claims, *args, **kwargs)
    return wrapper


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "online"}), 200


@app.route("/api/check-user", methods=["POST"])
def check_user():
    data = request.get_json(silent=True) or {}
    contact_type, canonical, variants = normalize_contact(data.get("contact"))
    if not contact_type:
        return error("Please enter a valid email address or 10-digit mobile number.", 400)

    resp = db("GET", {**contact_filter(variants), "select": "contact"})
    if resp is None:
        return error("Service temporarily unavailable.", 502)
    if resp.status_code != 200:
        app.logger.error("check-user query failed with status %s", resp.status_code)
        return error("Something went wrong. Please try again.", 500)

    records = resp.json()
    if records:
        return jsonify({"success": True, "exists": True, "contact": records[0].get("contact") or canonical}), 200
    return jsonify({"success": True, "exists": False, "contact": canonical}), 200


@app.route("/api/register", methods=["POST"])
def register_user():
    data = request.get_json(silent=True) or {}

    full_name = clean_field(data, "full_name")
    state = clean_field(data, "state")
    password = data.get("password") or ""
    contact_type, canonical, variants = normalize_contact(data.get("contact"))

    if not full_name:
        return error("Full name is required.", 400)
    if not contact_type:
        return error("Contact must be a valid email or 10-digit mobile number.", 400)
    if not isinstance(password, str) or not 6 <= len(password) <= 128:
        return error("Password must be between 6 and 128 characters.", 400)
    if not state:
        return error("State is required.", 400)

    existing = db("GET", {**contact_filter(variants), "select": "user_ui"})
    if existing is None:
        return error("Service temporarily unavailable.", 502)
    if existing.status_code == 200 and existing.json():
        return error(f"An account with this {contact_type} already exists. Please sign in.", 409)

    new_user = {field: clean_field(data, field) for field in PROFILE_FIELDS}
    new_user.update({
        "full_name": full_name,
        "state": state,
        "contact": canonical,
        "password": generate_password_hash(password, method="pbkdf2:sha256"),
        "is_active": True,
    })

    resp = db("POST", body=new_user)
    if resp is None:
        return error("Service temporarily unavailable.", 502)
    if resp.status_code == 409:
        return error(f"An account with this {contact_type} already exists. Please sign in.", 409)
    if resp.status_code not in (200, 201):
        app.logger.error("Registration insert failed with status %s", resp.status_code)
        return error("Could not create your account. Please try again.", 500)

    created = resp.json()
    created = created[0] if isinstance(created, list) and created else {}

    return jsonify({
        "success": True,
        "message": "Registration successful.",
        "token": make_token(created),
        "token_type": "Bearer",
        "expires_in_hours": JWT_HOURS,
        "user": sanitize_user(created),
    }), 201


@app.route("/api/login", methods=["POST"])
def login_user():
    data = request.get_json(silent=True) or {}
    password = data.get("password") or ""
    contact_type, _, variants = normalize_contact(data.get("contact"))

    if not contact_type or not isinstance(password, str) or not password:
        return error("Both contact and password are required.", 400)

    resp = db("GET", {**contact_filter(variants), "select": "*"})
    if resp is None:
        return error("Service temporarily unavailable.", 502)
    if resp.status_code != 200:
        app.logger.error("Login query failed with status %s", resp.status_code)
        return error("Something went wrong. Please try again.", 500)

    records = resp.json()
    if not records:
        return error("Invalid contact or password.", 401)

    user = records[0]
    stored = user.get("password") or ""

    if stored.startswith(("pbkdf2:", "scrypt:")):
        valid = check_password_hash(stored, password)
    else:
        legacy_hash = "sha256:" + hashlib.sha256(password.encode("utf-8")).hexdigest()
        valid = hmac.compare_digest(stored, legacy_hash)
        if valid and user.get("user_ui"):
            db(
                "PATCH",
                {"user_ui": f"eq.{user['user_ui']}"},
                {"password": generate_password_hash(password, method="pbkdf2:sha256")},
            )

    if not valid:
        return error("Invalid contact or password.", 401)
    if user.get("is_active") is False:
        return error("This account is currently inactive.", 403)

    profile = sanitize_user(user)
    return jsonify({
        "success": True,
        "message": f"Welcome back, {profile.get('full_name', 'Citizen')}!",
        "token": make_token(user),
        "token_type": "Bearer",
        "expires_in_hours": JWT_HOURS,
        "user": profile,
    }), 200


@app.route("/api/me", methods=["GET"])
@app.route("/api/profile", methods=["GET"])
@token_required
def get_profile(claims):
    resp = db("GET", {"user_ui": f"eq.{claims.get('user_ui')}", "select": "*"})
    if resp is None:
        return error("Service temporarily unavailable.", 502)
    if resp.status_code == 200 and resp.json():
        return jsonify({"success": True, "user": sanitize_user(resp.json()[0])}), 200
    return error("User profile not found.", 404)


@app.route("/api/profile", methods=["PUT"])
@token_required
def update_profile(claims):
    data = request.get_json(silent=True) or {}
    updates = {}
    for field in PROFILE_FIELDS:
        if field in data:
            updates[field] = clean_field(data, field)

    if not updates:
        return error("No valid profile fields provided to update.", 400)
    if "full_name" in updates and not updates["full_name"]:
        return error("Full name cannot be empty.", 400)

    updates["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    resp = db("PATCH", {"user_ui": f"eq.{claims.get('user_ui')}"}, updates)
    if resp is None:
        return error("Service temporarily unavailable.", 502)
    if resp.status_code not in (200, 204):
        app.logger.error("Profile update failed with status %s", resp.status_code)
        return error("Could not update your profile. Please try again.", 500)

    records = resp.json() if resp.text else []
    if not records:
        return error("User profile not found.", 404)
    return jsonify({
        "success": True,
        "message": "Profile updated successfully.",
        "user": sanitize_user(records[0]),
    }), 200


if __name__ == "__main__":
    app.run(
        host=env("FLASK_HOST", "127.0.0.1"),
        port=int(env("FLASK_PORT", "5000")),
        debug=env("FLASK_DEBUG", "False").lower() in ("true", "1", "yes"),
    )