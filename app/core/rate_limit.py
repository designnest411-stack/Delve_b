"""In-memory rate limiter; zero external Redis dependency."""

from __future__ import annotations

import time
from collections import defaultdict
from fastapi import HTTPException

# Thread-safe in-memory sliding window tracker: key -> list of timestamp floats
_rate_records: dict[str, list[float]] = defaultdict(list)


async def enforce_rate_limit(*, key: str, limit: int, window_seconds: int, message: str) -> None:
    """Enforce rate limits using an in-memory sliding window."""
    now = time.time()
    cutoff = now - window_seconds

    # Clean old timestamps
    current_calls = [t for t in _rate_records[key] if t > cutoff]

    if len(current_calls) >= limit:
        raise HTTPException(
            status_code=429,
            detail=message,
            headers={"Retry-After": str(window_seconds)},
        )

    current_calls.append(now)
    _rate_records[key] = current_calls

