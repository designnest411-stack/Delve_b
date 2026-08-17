"""Small distributed fixed-window rate limiter backed by Upstash Redis REST."""

from __future__ import annotations

import time

import httpx
from fastapi import HTTPException

from app.core.config import settings


async def enforce_rate_limit(*, key: str, limit: int, window_seconds: int, message: str) -> None:
    """Enforce rate limits using Upstash Redis. In development mode, no-ops gracefully."""
    if not settings.is_production:
        return

    if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
        raise HTTPException(status_code=503, detail="Rate limiting is not configured")

    bucket = int(time.time() // window_seconds)
    redis_key = f"delve:rate:{key}:{bucket}"
    headers = {"Authorization": f"Bearer {settings.upstash_redis_rest_token}"}
    commands = [["INCR", redis_key], ["EXPIRE", redis_key, str(window_seconds + 5), "NX"]]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{settings.upstash_redis_rest_url.rstrip('/')}/pipeline",
                headers=headers,
                json=commands,
            )
            response.raise_for_status()
        count = int(response.json()[0]["result"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Rate limiter temporarily unavailable") from exc

    if count > limit:
        raise HTTPException(status_code=429, detail=message, headers={"Retry-After": str(window_seconds)})
