"""Authentication and short-lived WebSocket tickets for hosted Delve."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from fastapi import HTTPException, Request, status

from app.core.config import settings

logger = logging.getLogger("delve.auth")


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str | None = None


_jwks_cache: tuple[float, dict[str, Any]] | None = None
_JWKS_CACHE_SECONDS = 3600


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


async def _get_jwks() -> dict[str, Any]:
    global _jwks_cache
    now = time.time()
    if _jwks_cache and _jwks_cache[0] > now:
        return _jwks_cache[1]
    if not settings.supabase_url:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json")
        response.raise_for_status()
        payload = response.json()
    _jwks_cache = (now + _JWKS_CACHE_SECONDS, payload)
    return payload


async def _decode_access_token(token: str) -> dict[str, Any]:
    try:
        if settings.supabase_jwt_secret:
            return jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience=settings.supabase_jwt_audience,
            )

        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        jwks = await _get_jwks()
        jwk = next((item for item in jwks.get("keys", []) if item.get("kid") == kid), None)
        if not jwk:
            raise jwt.InvalidTokenError("Signing key not found")
        key = jwt.PyJWK.from_dict(jwk).key
        return jwt.decode(
            token,
            key,
            algorithms=[header.get("alg", "RS256")],
            audience=settings.supabase_jwt_audience,
        )
    except Exception as exc:
        logger.warning("Token decoding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(request: Request) -> AuthenticatedUser:
    """Validate a Supabase access token; never trust a client supplied user id."""
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = await _decode_access_token(token)
    subject = str(claims.get("sub", "")).strip()
    if not subject:
        raise HTTPException(status_code=401, detail="Access token has no subject")
    return AuthenticatedUser(id=subject, email=claims.get("email"))


def create_ws_ticket(*, user_id: str, session_id: str, ttl_seconds: int = 60) -> str:
    """Create an opaque, short-lived ticket safe to place in a WS query string."""
    if not settings.ws_ticket_secret:
        raise HTTPException(status_code=503, detail="WebSocket authentication is not configured")
    payload = {
        "sub": user_id,
        "sid": session_id,
        "exp": int(time.time()) + ttl_seconds,
    }
    encoded = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        settings.ws_ticket_secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded}.{_b64url_encode(signature)}"


def verify_ws_ticket(ticket: str, *, session_id: str) -> AuthenticatedUser:
    if not settings.ws_ticket_secret:
        raise HTTPException(status_code=503, detail="WebSocket authentication is not configured")
    try:
        encoded, signature = ticket.split(".", 1)
        expected = hmac.new(
            settings.ws_ticket_secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(_b64url_decode(signature), expected):
            raise ValueError("Signature mismatch")
        payload = json.loads(_b64url_decode(encoded))
        if payload.get("sid") != session_id or int(payload.get("exp", 0)) < time.time():
            raise ValueError("Ticket expired or belongs to another session")
        user_id = str(payload.get("sub", "")).strip()
        if not user_id:
            raise ValueError("Ticket subject missing")
        return AuthenticatedUser(id=user_id)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid WebSocket ticket") from exc
