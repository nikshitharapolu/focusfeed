from __future__ import annotations
import hashlib, hmac, os, re, secrets, uuid
from typing import Any
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
from app.database import connection

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

class Credentials(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)

def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return f"pbkdf2_sha256$600000${salt.hex()}${digest.hex()}"

def _verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt, expected = stored.split("$", 3)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError): return False

def _email(value: str) -> str:
    value = value.strip().lower()
    if not EMAIL_RE.fullmatch(value): raise HTTPException(422, "Enter a valid email address")
    return value

def register(body: Credentials) -> dict[str, Any]:
    user_id, email = str(uuid.uuid4()), _email(body.email)
    try:
        with connection() as database:
            row = database.execute(
                "INSERT INTO users (id,email,password_hash) VALUES (%s,%s,%s) RETURNING id::text,email,created_at",
                (user_id, email, _hash_password(body.password)),
            ).fetchone()
    except Exception as exc:
        if getattr(exc, "sqlstate", None) == "23505":
            raise HTTPException(409, "An account with this email already exists") from exc
        raise
    return dict(row)

def authenticate(body: Credentials) -> dict[str, Any]:
    with connection() as database:
        row = database.execute("SELECT id::text,email,password_hash,created_at FROM users WHERE email=%s", (_email(body.email),)).fetchone()
    if not row or not _verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "Incorrect email or password")
    return {key: row[key] for key in ("id", "email", "created_at")}

def current_user(request: Request) -> dict[str, Any]:
    user_id = request.session.get("user_id")
    if not user_id: raise HTTPException(401, "Sign in to continue")
    with connection() as database:
        row = database.execute("SELECT id::text,email,created_at FROM users WHERE id=%s", (user_id,)).fetchone()
    if not row:
        request.session.clear(); raise HTTPException(401, "Session is no longer valid")
    return dict(row)

def session_secret() -> str:
    return os.getenv("FOCUSFEED_SESSION_SECRET", "local-development-secret-change-before-deployment")
