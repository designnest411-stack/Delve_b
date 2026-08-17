"""Per-research-call budget accounting that prevents runaway agent loops."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, field

from app.core.config import settings


@dataclass
class ResearchLLMBudget:
    session_id: str
    owner_id: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_budget: contextvars.ContextVar[ResearchLLMBudget | None] = contextvars.ContextVar("delve_llm_budget", default=None)
_global_semaphore: asyncio.Semaphore | None = None


def activate_budget(session_id: str, owner_id: str) -> contextvars.Token:
    return _budget.set(ResearchLLMBudget(session_id=session_id, owner_id=owner_id))


def deactivate_budget(token: contextvars.Token) -> None:
    _budget.reset(token)


def current_budget() -> ResearchLLMBudget | None:
    return _budget.get()


async def reserve_llm_call() -> asyncio.Semaphore | None:
    """Reserve a call before it reaches Gemini API; shared across summary tasks."""
    budget = current_budget()
    if budget:
        async with budget.lock:
            if budget.calls >= settings.max_llm_calls_per_job:
                raise RuntimeError("Research job reached its LLM-call safety limit")
            budget.calls += 1
    global _global_semaphore
    if _global_semaphore is None:
        _global_semaphore = asyncio.Semaphore(settings.max_llm_calls_in_flight)
    await _global_semaphore.acquire()
    return _global_semaphore


async def record_usage(usage: dict) -> None:
    budget = current_budget()
    if not budget:
        return
    async with budget.lock:
        budget.input_tokens += int(usage.get("input_tokens", 0) or 0)
        budget.output_tokens += int(usage.get("output_tokens", 0) or 0)
        budget.cache_read_tokens += int(usage.get("cache_read_input_tokens", 0) or 0)
