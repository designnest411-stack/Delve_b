"""QStash publisher for free-tier, durable background research execution."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings


async def enqueue_research_job(job_id: str) -> str:
    """Publish a single idempotent job reference; never put research content in QStash."""
    if not settings.qstash_token or not settings.public_api_base_url:
        raise RuntimeError("QStash deployment configuration is missing")
    if not settings.job_dispatch_secret:
        raise RuntimeError("JOB_DISPATCH_SECRET is missing")
    destination = f"{settings.public_api_base_url.rstrip('/')}/research/internal/jobs/{job_id}/run"
    headers = {
        "Authorization": f"Bearer {settings.qstash_token}",
        "Content-Type": "application/json",
        "Upstash-Timeout": "14m",
        "Upstash-Retries": "2",
        "Upstash-Retry-Delay": "pow(2, retried) * 60000",
        "Upstash-Forward-Content-Type": "application/json",
        "Upstash-Forward-X-Delve-Job-Secret": settings.job_dispatch_secret,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{settings.qstash_url.rstrip('/')}/v2/publish/{destination}",
            headers=headers,
            json={"job_id": job_id},
        )
    if response.status_code >= 400:
        raise RuntimeError(f"QStash enqueue failed ({response.status_code}): {response.text[:300]}")
    return str(response.json().get("messageId", ""))
