"""
Delve Research Routes
──────────────────────
POST /research/start                        – Start a new research session
POST /research/{id}/cancel                  – Request cancellation
POST /research/{id}/retry                   – Retry a failed/cancelled session
POST /research/{id}/resume                  – Alias for retry
DELETE /research/{id}                       – Delete a session
GET  /research/{id}/status                  – Session status
GET  /research/{id}/detail                  – Rich session metadata
GET  /research/{id}/paper                   – Completed paper content
GET  /research/{id}/chat                    – Chat history
POST /research/{id}/chat                    – Chat with session context
GET  /research/{id}/timeline                – Timeline events
GET  /research/{id}/export                  – Full export bundle (MD/BibTeX/JSON)
GET  /research/{id}/paper.pdf               – PDF download
POST /research/{id}/ws-ticket               – Short-lived WebSocket auth ticket
GET  /research/sessions/list                – List all sessions
POST /research/internal/jobs/{id}/run       – QStash job runner (internal)
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.core.agents import _debate_snippet
from app.core.config import settings
from app.core.auth import AuthenticatedUser, create_ws_ticket, get_current_user
from app.core.llm_budget import activate_budget, current_budget, deactivate_budget
from app.core.llm_client import llm_client
from app.core.qstash import enqueue_research_job
from app.core.rate_limit import enforce_rate_limit
from app.core.supabase import supabase_repository
from app.services.pdf_export import generate_research_pdf
from app.services.supabase_vectors import query_uploaded_documents

logger = logging.getLogger("delve.api.research")

router = APIRouter(prefix="/research", tags=["research"])

ALLOWED_PAPER_FORMATS = {"academic", "ieee", "apa", "acm", "mla"}
ALLOWED_DEPTHS = {"quick", "standard", "deep"}
ALLOWED_SOURCES = {"arxiv", "semantic_scholar", "openalex", "crossref", "github_repo", "web_tavily"}
SAFE_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ── Pydantic Models ───────────────────────────────────────────────────────

class StartResearchRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    uploaded_paper_ids: list[str] = Field(default_factory=list)
    max_debate_rounds: int | None = Field(default=None, ge=0, le=5)
    strict_mode: bool | None = None
    paper_format: Literal["academic", "ieee", "apa", "acm", "mla"] | None = "academic"
    depth: Literal["quick", "standard", "deep"] | None = "standard"
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    include_sources: list[str] | None = None
    exclude_sources: list[str] | None = None

    @field_validator("topic")
    @classmethod
    def _clean_topic(cls, value: str) -> str:
        topic = value.strip()
        if not topic:
            raise ValueError("Topic cannot be empty")
        return topic

    @field_validator("uploaded_paper_ids")
    @classmethod
    def _validate_uploaded_ids(cls, value: list[str]) -> list[str]:
        bad = [item for item in value if not SAFE_SESSION_ID_RE.fullmatch(item)]
        if bad:
            raise ValueError("Uploaded paper ids may only contain letters, numbers, underscores, and hyphens")
        return value

    @field_validator("include_sources", "exclude_sources")
    @classmethod
    def _validate_sources(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        unknown = sorted({source for source in value if source not in ALLOWED_SOURCES})
        if unknown:
            raise ValueError(f"Unknown sources: {', '.join(unknown)}")
        return value


class StartResearchResponse(BaseModel):
    session_id: str
    message: str


class SessionStatus(BaseModel):
    session_id: str
    status: str
    current_step: str
    paper_available: bool


class SessionActionResponse(BaseModel):
    session_id: str
    status: str
    message: str


class ResearchChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ResearchChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class ResearchChatResponse(BaseModel):
    session_id: str
    answer: str
    history: list[ResearchChatMessage]


# ── In-memory state (cache for active / recently-run sessions) ────────────

# active_sessions is the runtime cache. Supabase is the authoritative store.
active_sessions: dict[str, dict[str, Any]] = {}
ws_connections: dict[str, list[WebSocket]] = {}

# Debounced Supabase persist tasks (keyed by session_id)
_pending_persist_tasks: dict[str, asyncio.Task] = {}


# ── Session Validation ────────────────────────────────────────────────────

def _safe_session_segment(session_id: str) -> str:
    if not session_id or not SAFE_SESSION_ID_RE.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    return session_id


# ── Supabase Persistence ──────────────────────────────────────────────────

async def _persist_session(session_id: str) -> None:
    """Upsert a single session to Supabase."""
    session = active_sessions.get(session_id)
    if not session or "owner_id" not in session:
        return
    try:
        await supabase_repository.upsert_session(session_id, session)
    except Exception as exc:
        logger.error("Failed to persist session %s: %s", session_id, exc)


async def _persist_session_debounced(session_id: str, delay: float = 0.8) -> None:
    """Debounced Supabase upsert — cancels any in-flight write before scheduling."""
    existing = _pending_persist_tasks.get(session_id)
    if existing and not existing.done():
        existing.cancel()

    async def _job() -> None:
        try:
            await asyncio.sleep(delay)
            await _persist_session(session_id)
        except asyncio.CancelledError:
            return
        finally:
            if _pending_persist_tasks.get(session_id) is asyncio.current_task():
                _pending_persist_tasks.pop(session_id, None)

    _pending_persist_tasks[session_id] = asyncio.create_task(_job())


async def flush_pending_persists() -> None:
    """Cancel all debounced tasks and force one final flush. Called on shutdown."""
    for task in list(_pending_persist_tasks.values()):
        if not task.done():
            task.cancel()
    _pending_persist_tasks.clear()
    # Flush all in-memory sessions to Supabase
    for session_id in list(active_sessions.keys()):
        await _persist_session(session_id)


# ── Timeline / Broadcast ──────────────────────────────────────────────────

def _timeline_event(session_id: str, message: dict[str, Any]) -> None:
    session = active_sessions.get(session_id)
    if not session:
        return
    timeline = session.setdefault("timeline", [])
    timeline.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": message.get("type", "status"),
        "message": message.get("message", ""),
        "data": message.get("data", {}),
    })
    if len(timeline) > 700:
        del timeline[: len(timeline) - 700]


def _infer_step(message: str) -> str:
    lower = message.lower()
    if "planning" in lower or "query" in lower:
        return "planner"
    if "searching academic databases" in lower or "retrieved" in lower:
        return "retrieval"
    if "summariz" in lower:
        return "summarizer"
    if "draft" in lower:
        return "proposer"
    if "peer review" in lower:
        return "critic"
    if "gap" in lower:
        return "gap_analysis"
    if "assembling" in lower or "paper complete" in lower:
        return "paper_architect"
    return ""


async def broadcast_to_session(session_id: str, message: dict[str, Any]) -> None:
    session = active_sessions.get(session_id)
    if session:
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        metrics = session.setdefault("metrics", {})
        if message.get("type") == "status":
            data = message.setdefault("data", {})
            step = str((data or {}).get("node", "")).strip() or _infer_step(str(message.get("message", "")))
            if step:
                data["node"] = step
                session["current_step"] = step
            if "token_estimate" in data:
                metrics["token_estimate"] = int(data.get("token_estimate") or 0)
                metrics["cost_estimate_usd"] = round(metrics["token_estimate"] * 0.00000035, 4)
        _timeline_event(session_id, message)

    connections = ws_connections.get(session_id, [])
    dead: list[WebSocket] = []
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)

    await _persist_session_debounced(session_id)


async def _emit_node_status(session_id: str, node: str, message: str, data: dict[str, Any] | None = None) -> None:
    payload_data = {"node": node}
    if data:
        payload_data.update(data)
    await broadcast_to_session(session_id, {
        "type": "status",
        "message": message,
        "data": payload_data,
    })


def _session_elapsed_seconds(session: dict[str, Any]) -> int:
    started = session.get("started_at")
    if not started:
        return 0
    try:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end_raw = session.get("finished_at")
        end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00")) if end_raw else datetime.now(timezone.utc)
        return max(0, int((end_dt - start_dt).total_seconds()))
    except Exception:
        return 0


# ── Export Bundle ─────────────────────────────────────────────────────────

def _build_export_bundle(session_id: str, session: dict[str, Any]) -> dict[str, Any]:
    result = session.get("result", {})
    paper = str(result.get("final_paper", "") or "")
    analysis = str(result.get("research_analysis", "") or "")
    final_draft = str(result.get("final_draft", paper) or "")
    bibliography = result.get("bibliography", [])
    evidence = {
        "citation_quality": result.get("citation_quality", {}),
        "citation_verification": result.get("citation_verification", {}),
        "claim_to_evidence_map": result.get("claim_to_evidence_map", []),
        "duplicate_clusters": result.get("duplicate_clusters", []),
        "source_counts": result.get("source_counts", {}),
    }
    bibtex_entries = []
    for idx, b in enumerate(bibliography, start=1):
        key = f"delve{idx}"
        bibtex_entries.append(
            f"@article{{{key}, title={{{b.get('title','')}}}, author={{{b.get('authors','')}}}, "
            f"year={{{b.get('year','')}}}, url={{{b.get('url','')}}}, doi={{{b.get('doi','')}}}}}"
        )
    return {
        "session_id": session_id,
        "markdown": paper,
        "analysis_markdown": analysis,
        "final_draft_markdown": final_draft,
        "bibtex": "\n\n".join(bibtex_entries),
        "evidence_json": evidence,
        "chat_history": session.get("chat_history", []),
        "timeline": session.get("timeline", []),
    }


# ── Pipeline Runner ───────────────────────────────────────────────────────

async def run_research_pipeline(
    session_id: str,
    topic: str,
    uploaded_paper_ids: list[str],
    controls: dict[str, Any],
) -> None:
    budget_token = None
    try:
        session = active_sessions[session_id]
        budget_token = activate_budget(session_id, str(session.get("owner_id", "")))
        session["status"] = "running"
        session["current_step"] = "planner"
        session["started_at"] = session.get("started_at", datetime.now(timezone.utc).isoformat())
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        session["finished_at"] = None
        session["controls"] = controls
        session["run_id"] = session.get("run_id") or str(uuid.uuid4())
        await _persist_session(session_id)

        async def ws_callback(msg: dict[str, Any]) -> None:
            await broadcast_to_session(session_id, msg)

        initial_state: dict[str, Any] = {
            "topic": topic,
            "uploaded_paper_ids": uploaded_paper_ids,
            "session_id": session_id,
            "owner_id": session.get("owner_id", ""),
            "retrieved_papers": [],
            "vector_results": [],
            "paper_summaries": {},
            "debate_log": [],
            "identified_gaps": [],
            "bibliography": [],
            "strict_mode": bool(controls.get("strict_mode", settings.strict_synthesis_mode)),
            "max_debate_rounds": int(controls.get("max_debate_rounds", settings.max_debate_rounds)),
            "paper_format": str(controls.get("paper_format", "academic")),
            "planner_constraints": {
                "depth": controls.get("depth", "standard"),
                "year_from": controls.get("year_from"),
                "include_sources": controls.get("include_sources", []),
                "exclude_sources": controls.get("exclude_sources", []),
            },
        }

        # Each run gets its own checkpointer. Use SQLite for persistence across retries.
        config = {"configurable": {"thread_id": session_id, "ws_callback": ws_callback}}

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from app.core.graph import build_research_graph

        await _emit_node_status(session_id, "planner", f"Starting research on: {topic}")

        # Use async SQLite checkpointer for persistent state across retries
        checkpoint_db = Path(tempfile.gettempdir()) / "delve_checkpoints.db"
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as checkpointer:
            graph = build_research_graph(checkpointer=checkpointer)
            
            current_state = await graph.aget_state(config)
            if current_state.values:
                logger.info(f"Resuming session {session_id} from checkpoint")
                final_state = await graph.ainvoke(None, config=config)
            else:
                final_state = await graph.ainvoke(initial_state, config=config)

        debate_log = final_state.get("debate_log", [])
        proposer_entries = [d for d in debate_log if isinstance(d, str) and d.startswith("[PROPOSER]")]
        critic_entries = [d for d in debate_log if isinstance(d, str) and d.startswith("[CRITIC]")]

        result = {
            "final_paper": final_state.get("final_paper_markdown", ""),
            "research_analysis": final_state.get("research_analysis_markdown", ""),
            "final_draft": final_state.get("final_draft_markdown", final_state.get("final_paper_markdown", "")),
            "gaps": final_state.get("identified_gaps", []),
            "gap_candidates": final_state.get("generated_gap_candidates", []),
            "gap_critique": final_state.get("gap_critique", ""),
            "bibliography": final_state.get("bibliography", []),
            "debate_log": debate_log,
            "paper_summaries": final_state.get("paper_summaries", {}),
            "literature_review": final_state.get("literature_review_draft", ""),
            "cross_paper_analysis": final_state.get("cross_paper_analysis", {}),
            "citation_quality": final_state.get("citation_quality", {}),
            "citation_verification": final_state.get("citation_verification", {}),
            "source_counts": final_state.get("source_counts", {}),
            "duplicate_clusters": final_state.get("duplicate_clusters", []),
            "claim_to_evidence_map": final_state.get("claim_to_evidence_map", []),
            "debate_rounds": final_state.get("debate_rounds", len(critic_entries)),
            "verified_citations": final_state.get("verified_citations", 0),
            "paper_format": controls.get("paper_format", "academic"),
            "format_compliance": final_state.get("format_compliance", {}),
        }

        budget = current_budget()
        measured_tokens = (budget.input_tokens + budget.output_tokens) if budget else 0
        measured_cost = 0.0  # Google Gemini Free Tier ($0.00 USD)

        session.update({
            "status": "complete",
            "current_step": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "metrics": {
                "elapsed_seconds": _session_elapsed_seconds(session),
                "token_estimate": measured_tokens,
                "input_tokens": budget.input_tokens if budget else 0,
                "output_tokens": budget.output_tokens if budget else 0,
                "cache_read_tokens": budget.cache_read_tokens if budget else 0,
                "llm_calls": budget.calls if budget else 0,
                "cost_estimate_usd": measured_cost,
                "error_count": 0,
            },
        })

        if budget:
            await supabase_repository.insert_llm_usage({
                "session_id": session_id,
                "owner_id": session["owner_id"],
                "agent": "pipeline",
                "model": settings.llm_model,
                "input_tokens": budget.input_tokens,
                "output_tokens": budget.output_tokens,
                "cache_read_tokens": budget.cache_read_tokens,
                "estimated_cost_usd": 0,
            })

        # Replay debate entries over WebSocket for any connected clients
        for idx, _entry in enumerate(proposer_entries, start=1):
            await broadcast_to_session(session_id, {
                "type": "debate",
                "message": f"Debate round {idx}: proposer draft prepared",
                "data": {
                    "round": idx,
                    "speaker": "proposer",
                    "node": "proposer",
                    "snippet": _debate_snippet(_entry),
                },
            })
            if idx <= len(critic_entries):
                await broadcast_to_session(session_id, {
                    "type": "debate",
                    "message": f"Debate round {idx}: critic feedback delivered",
                    "data": {
                        "round": idx,
                        "speaker": "critic",
                        "node": "critic",
                        "snippet": _debate_snippet(critic_entries[idx - 1]),
                    },
                })

        await broadcast_to_session(session_id, {
            "type": "complete",
            "message": "Research complete! Paper is ready.",
            "data": {
                "paper_length": len(result["final_paper"]),
                "num_gaps": len(result["gaps"]),
                "num_references": len(result["bibliography"]),
                "node": "complete",
                "token_estimate": session.get("metrics", {}).get("token_estimate", 0),
                "cost_estimate_usd": session.get("metrics", {}).get("cost_estimate_usd", 0),
                "debate_rounds": result.get("debate_rounds", 0),
                "verified_citations": result.get("verified_citations", 0),
                "format_compliance_score": (result.get("format_compliance", {}) or {}).get("score", 0.0),
            },
        })

    except asyncio.CancelledError:
        session = active_sessions.get(session_id)
        if session:
            session.update({
                "status": "cancelled",
                "current_step": "cancelled",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
        await broadcast_to_session(session_id, {
            "type": "status",
            "message": "Research cancelled.",
            "data": {"node": "cancelled"},
        })
        raise
    except Exception as e:
        logger.error("Research pipeline failed for session %s: %s", session_id, e, exc_info=True)
        session = active_sessions.get(session_id)
        if session:
            session.update({
                "status": "error",
                "current_step": "error",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            })
            session.setdefault("metrics", {})["error_count"] = (
                int(session["metrics"].get("error_count", 0)) + 1
            )
        await broadcast_to_session(session_id, {
            "type": "error",
            "message": f"Research failed: {str(e)[:200]}",
            "data": {"node": "error"},
        })
    finally:
        if budget_token is not None:
            deactivate_budget(budget_token)
        try:
            await _persist_session(session_id)
        except Exception as exc:
            logger.error("Final persist failed for session %s: %s", session_id, exc)


# ── Session Helpers ───────────────────────────────────────────────────────

async def _get_owned_session(session_id: str, user: AuthenticatedUser) -> dict[str, Any]:
    """Load a session from in-memory cache or Supabase, enforcing ownership."""
    _safe_session_segment(session_id)
    session = active_sessions.get(session_id)
    if session is not None:
        if session.get("owner_id") != user.id:
            raise HTTPException(status_code=403, detail="You do not have access to this session")
        return session
    session = await supabase_repository.get_session(session_id, user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not settings.is_production and session.get("status") in {"running", "queued"}:
        session["status"] = "error"
        session["error"] = "Session was interrupted due to server restart. Please retry."
        try:
            await supabase_repository.upsert_session(session_id, session)
        except Exception:
            pass

    active_sessions[session_id] = session
    ws_connections.setdefault(session_id, [])
    return session


# ── REST Endpoints ────────────────────────────────────────────────────────

@router.post("/start", response_model=StartResearchResponse)
async def start_research(
    request: StartResearchRequest,
    user: AuthenticatedUser = Depends(get_current_user),
) -> StartResearchResponse:
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    # Check lifetime quota (5 free papers per account)
    has_quota = await supabase_repository.check_user_quota(user.id)
    if not has_quota:
        quota_info = await supabase_repository.get_user_quota_info(user.id)
        raise HTTPException(
            status_code=403,
            detail=f"Free paper limit reached. You have generated {quota_info['papers_generated']} paper(s). Upgrade for unlimited access."
        )

    await enforce_rate_limit(
        key=f"research-day:{user.id}", limit=settings.max_research_jobs_per_day,
        window_seconds=86400, message="Daily research-job limit reached",
    )
    if await supabase_repository.count_active_jobs(user.id) >= settings.max_concurrent_jobs_per_user:
        raise HTTPException(status_code=429, detail="You already have a research job in progress")

    if request.uploaded_paper_ids:
        documents = await supabase_repository.get_documents(request.uploaded_paper_ids, user.id)
        if len(documents) != len(set(request.uploaded_paper_ids)):
            raise HTTPException(status_code=403, detail="One or more uploaded documents do not belong to you")
        if any(doc.get("status") != "ready" for doc in documents):
            raise HTTPException(status_code=409, detail="Uploaded documents are still processing or failed")

    session_id = str(uuid.uuid4())
    requested_rounds = (
        int(request.max_debate_rounds)
        if request.max_debate_rounds is not None
        else int(settings.max_debate_rounds)
    )
    controls = {
        "max_debate_rounds": max(0, min(5, requested_rounds)),
        "strict_mode": bool(request.strict_mode) if request.strict_mode is not None else bool(settings.strict_synthesis_mode),
        "paper_format": (request.paper_format or "academic").strip().lower(),
        "depth": request.depth or "standard",
        "year_from": request.year_from,
        "include_sources": request.include_sources or [],
        "exclude_sources": request.exclude_sources or [],
    }
    active_sessions[session_id] = {
        "owner_id": user.id,
        "status": "queued",
        "current_step": "queued",
        "topic": request.topic,
        "uploaded_paper_ids": request.uploaded_paper_ids,
        "controls": controls,
        "metrics": {"elapsed_seconds": 0, "token_estimate": 0, "cost_estimate_usd": 0, "error_count": 0},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "run_id": str(uuid.uuid4()),
        "timeline": [],
        "chat_history": [],
    }
    ws_connections[session_id] = []
    await _persist_session(session_id)

    job = await supabase_repository.create_job(session_id, user.id)
    if not settings.is_production:
        # In dev mode, QStash cannot reach localhost — run the pipeline directly
        # in a background task instead of going through QStash.
        asyncio.create_task(_run_dev_job(str(job["id"]), session_id, user.id))
    else:
        try:
            await enqueue_research_job(str(job["id"]))
        except Exception as exc:
            active_sessions[session_id].update({
                "status": "error",
                "current_step": "error",
                "error": "Could not queue research job",
            })
            await _persist_session(session_id)
            await supabase_repository.finish_job(str(job["id"]), "error", str(exc)[:300])
            raise HTTPException(status_code=503, detail="Research queue is temporarily unavailable") from exc

    return StartResearchResponse(session_id=session_id, message="Research queued")


async def _run_dev_job(job_id: str, session_id: str, owner_id: str) -> None:
    """Dev-only: run a research job in-process without QStash."""
    job = await supabase_repository.claim_job(job_id)
    if not job:
        return
    session = active_sessions.get(session_id) or await supabase_repository.get_session(session_id, owner_id)
    if not session or session.get("status") == "cancelled":
        await supabase_repository.finish_job(job_id, "cancelled")
        return
    active_sessions[session_id] = session
    ws_connections.setdefault(session_id, [])
    await run_research_pipeline(
        session_id=session_id,
        topic=session.get("topic", ""),
        uploaded_paper_ids=session.get("uploaded_paper_ids", []),
        controls=session.get("controls", {}),
    )
    final_status = active_sessions.get(session_id, {}).get("status", "error")
    await supabase_repository.finish_job(job_id, "complete" if final_status == "complete" else final_status)



@router.post("/internal/jobs/{job_id}/run")
async def run_queued_research_job(job_id: str, request: Request) -> dict[str, str]:
    """QStash-only endpoint. Verifies the dispatch secret, claims the job, then runs the pipeline."""
    supplied = request.headers.get("X-Delve-Job-Secret", "")
    if not settings.job_dispatch_secret or not hmac.compare_digest(supplied, settings.job_dispatch_secret):
        raise HTTPException(status_code=401, detail="Invalid job dispatcher credentials")
    job = await supabase_repository.claim_job(job_id)
    if not job:
        return {"status": "already-claimed"}
    session_id = str(job["session_id"])
    owner_id = str(job["owner_id"])
    session = await supabase_repository.get_session(session_id, owner_id)
    if not session or session.get("status") == "cancelled":
        await supabase_repository.finish_job(job_id, "cancelled")
        return {"status": "cancelled"}
    active_sessions[session_id] = session
    ws_connections.setdefault(session_id, [])
    await run_research_pipeline(
        session_id=session_id,
        topic=session.get("topic", ""),
        uploaded_paper_ids=session.get("uploaded_paper_ids", []),
        controls=session.get("controls", {}),
    )
    final_status = active_sessions.get(session_id, {}).get("status", "error")
    await supabase_repository.finish_job(job_id, "complete" if final_status == "complete" else final_status)
    return {"status": final_status}


@router.post("/{session_id}/cancel", response_model=SessionActionResponse)
async def cancel_session(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> SessionActionResponse:
    session = await _get_owned_session(session_id, user)
    session.update({
        "status": "cancelled",
        "current_step": "cancelled",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    await _persist_session(session_id)
    return SessionActionResponse(session_id=session_id, status="cancelled", message="Cancellation requested")


@router.post("/{session_id}/retry", response_model=SessionActionResponse)
async def retry_session(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> SessionActionResponse:
    session = await _get_owned_session(session_id, user)

    # In dev or on retry, cancel any previous orphaned jobs for this session
    try:
        await supabase_repository._rest(
            "PATCH", "research_jobs",
            params={"session_id": f"eq.{session_id}", "status": "in.(queued,running)"},
            payload={"status": "cancelled", "error": "Superseded by retry"},
            prefer="return=minimal",
        )
    except Exception:
        pass

    if await supabase_repository.count_active_jobs(user.id) >= settings.max_concurrent_jobs_per_user:
        raise HTTPException(status_code=429, detail="You already have a research job in progress")

    # Keep existing run_id, timeline, and chat_history to resume from checkpoint
    session.update({
        "status": "queued",
        "current_step": "queued",
        "error": None,
        "finished_at": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        # Preserve run_id, timeline, chat_history for checkpoint resumption
    })
    await _persist_session(session_id)
    job = await supabase_repository.create_job(session_id, user.id)
    if not settings.is_production:
        asyncio.create_task(_run_dev_job(str(job["id"]), session_id, user.id))
    else:
        try:
            await enqueue_research_job(str(job["id"]))
        except Exception as exc:
            session.update({"status": "error", "error": "Could not queue retry"})
            await _persist_session(session_id)
            await supabase_repository.finish_job(str(job["id"]), "error", str(exc)[:300])
            raise HTTPException(status_code=503, detail="Research queue temporarily unavailable") from exc
    return SessionActionResponse(session_id=session_id, status="queued", message="Retry queued")


@router.post("/{session_id}/resume", response_model=SessionActionResponse)
async def resume_session(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> SessionActionResponse:
    session = await _get_owned_session(session_id, user)
    if session.get("status") not in {"cancelled", "error"}:
        raise HTTPException(status_code=400, detail="Only cancelled/error sessions can be resumed")
    return await retry_session(session_id, user)


@router.delete("/{session_id}", response_model=SessionActionResponse)
async def delete_session(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> SessionActionResponse:
    await _get_owned_session(session_id, user)
    active_sessions.pop(session_id, None)
    ws_connections.pop(session_id, None)
    await supabase_repository.delete_session(session_id, user.id)
    return SessionActionResponse(session_id=session_id, status="deleted", message="Session deleted")


@router.get("/{session_id}/status", response_model=SessionStatus)
async def get_session_status(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> SessionStatus:
    session = await _get_owned_session(session_id, user)
    return SessionStatus(
        session_id=session_id,
        status=session.get("status", "unknown"),
        current_step=session.get("current_step", "unknown"),
        paper_available=bool(session.get("result", {}).get("final_paper")),
    )


@router.post("/{session_id}/ws-ticket")
async def create_session_ws_ticket(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, str]:
    """Return a 60-second HMAC ticket for WebSocket auth (never expose Supabase JWT in WS URL)."""
    await _get_owned_session(session_id, user)
    return {"ticket": create_ws_ticket(user_id=user.id, session_id=session_id), "expires_in": "60"}


@router.get("/{session_id}/detail")
async def get_session_detail(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    session = await _get_owned_session(session_id, user)
    metrics = session.get("metrics", {})
    return {
        "session_id": session_id,
        "topic": session.get("topic", ""),
        "status": session.get("status", "unknown"),
        "current_step": session.get("current_step", "unknown"),
        "started_at": session.get("started_at"),
        "finished_at": session.get("finished_at"),
        "elapsed_seconds": _session_elapsed_seconds(session),
        "source_counts": session.get("result", {}).get("source_counts", {}),
        "duplicate_clusters": session.get("result", {}).get("duplicate_clusters", []),
        "error": session.get("error"),
        "error_badge": "none" if not session.get("error") else "present",
        "token_estimate": metrics.get("token_estimate", 0),
        "cost_estimate_usd": metrics.get("cost_estimate_usd", 0),
        "controls": session.get("controls", {}),
        "uploaded_paper_ids": session.get("uploaded_paper_ids", []),
        "resource_files": [],
    }


@router.get("/{session_id}/paper")
async def get_paper(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    session = await _get_owned_session(session_id, user)
    if session.get("status") != "complete":
        raise HTTPException(status_code=400, detail=f"Session is {session.get('status')}, paper not ready")
    result = session.get("result", {})
    return {
        "session_id": session_id,
        "paper": str(result.get("final_paper", "") or ""),
        "analysis": str(result.get("research_analysis", "") or ""),
        "final_draft": str(result.get("final_draft", result.get("final_paper", "")) or ""),
        "gaps": result.get("gaps", []),
        "bibliography": result.get("bibliography", []),
        "debate_log": result.get("debate_log", []),
        "timeline": session.get("timeline", []),
        "cross_paper_analysis": result.get("cross_paper_analysis", {}),
        "gap_candidates": result.get("gap_candidates", []),
        "gap_critique": result.get("gap_critique", ""),
        "citation_quality": result.get("citation_quality", {}),
        "citation_verification": result.get("citation_verification", {}),
        "claim_to_evidence_map": result.get("claim_to_evidence_map", []),
        "source_counts": result.get("source_counts", {}),
        "duplicate_clusters": result.get("duplicate_clusters", []),
        "debate_rounds": result.get("debate_rounds", 0),
        "verified_citations": result.get("verified_citations", 0),
        "paper_format": result.get("paper_format", session.get("controls", {}).get("paper_format", "academic")),
        "format_compliance": result.get("format_compliance", {}),
    }


@router.get("/{session_id}/chat")
async def get_chat_history(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    session = await _get_owned_session(session_id, user)
    return {"session_id": session_id, "history": session.get("chat_history", [])}


@router.post("/{session_id}/chat", response_model=ResearchChatResponse)
async def chat_with_research(
    session_id: str,
    request: ResearchChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResearchChatResponse:
    await enforce_rate_limit(
        key=f"chat:{user.id}", limit=20, window_seconds=3600,
        message="Hourly research-chat limit reached",
    )
    session = await _get_owned_session(session_id, user)
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history = session.setdefault("chat_history", [])
    now_iso = datetime.now(timezone.utc).isoformat()
    history.append({"role": "user", "content": user_message, "timestamp": now_iso})

    recent_turns = history[-8:]
    conversation_text = "\n".join(
        f"{m.get('role', 'user').upper()}: {str(m.get('content', '')).strip()[:700]}"
        for m in recent_turns
        if str(m.get("content", "")).strip()
    )

    # RAG over uploaded documents via Supabase pgvector
    rag_context = ""
    uploaded_ids = session.get("uploaded_paper_ids", [])
    if uploaded_ids:
        chunks = await query_uploaded_documents(
            owner_id=user.id, document_ids=uploaded_ids, query=user_message, n_results=10,
        )
        if chunks:
            rag_context = "Relevant excerpts from uploaded documents:\n"
            for c in chunks:
                rag_context += f"- {c['text']}\n"

    result = session.get("result", {}) or {}
    context_payload = {
        "topic": session.get("topic", ""),
        "status": session.get("status", "unknown"),
        "paper_ready": bool(result.get("final_paper")),
        "literature_review": str(result.get("literature_review", ""))[:5000],
        "analysis": str(result.get("research_analysis", ""))[:5000],
        "cross_paper_analysis": result.get("cross_paper_analysis", {}),
        "gaps": result.get("gaps", []),
        "source_counts": result.get("source_counts", {}),
        "format_compliance": result.get("format_compliance", {}),
    }

    system_prompt = (
        "You are Delve Research Discussion Assistant. You answer questions about the selected research session. "
        "Stay grounded in the provided session context and the relevant RAG document excerpts. Be concrete and analytical. "
        "If information is missing in session context, clearly say it is not available yet."
    )
    user_prompt = (
        "Session context:\n"
        f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n\n"
        f"{rag_context}\n\n"
        "Recent conversation:\n"
        f"{conversation_text}\n\n"
        "User question:\n"
        f"{user_message}\n\n"
        "Respond with a practical, concise answer focused on this research."
    )

    answer = await llm_client.generate_with_system(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_output_tokens=1500,
        enable_thinking=True,
    )
    cleaned_answer = (answer or "").strip() or "I could not generate a response for this question."
    history.append({
        "role": "assistant",
        "content": cleaned_answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    if len(history) > 80:
        del history[: len(history) - 80]

    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    await _persist_session_debounced(session_id, delay=0.2)

    return ResearchChatResponse(
        session_id=session_id,
        answer=cleaned_answer,
        history=[
            ResearchChatMessage(
                role=str(m.get("role", "assistant")),
                content=str(m.get("content", "")),
                timestamp=str(m.get("timestamp", "")),
            )
            for m in history
        ],
    )


@router.get("/{session_id}/timeline")
async def get_timeline(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    session = await _get_owned_session(session_id, user)
    return {"session_id": session_id, "timeline": session.get("timeline", [])}


@router.get("/{session_id}/export")
async def export_session_bundle(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    session = await _get_owned_session(session_id, user)
    if session.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Session not complete")
    return _build_export_bundle(session_id, session)


@router.get("/{session_id}/paper.pdf")
async def download_paper_pdf(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> FileResponse:
    session = await _get_owned_session(session_id, user)
    if session.get("status") != "complete":
        raise HTTPException(status_code=400, detail=f"Session is {session.get('status')}, paper not ready")

    result = session.get("result", {}) or {}
    analysis = str(result.get("research_analysis", "") or "")
    final_draft = str(result.get("final_draft", result.get("final_paper", "")) or "")
    if not final_draft and not analysis:
        raise HTTPException(status_code=404, detail="No paper content available for PDF export")

    # Write to an isolated temp directory; ephemeral on Render which is intentional.
    tmp_dir = Path(tempfile.mkdtemp(prefix="delve_pdf_"))
    out_path = tmp_dir / "delve_paper.pdf"
    await asyncio.to_thread(
        generate_research_pdf,
        topic=session.get("topic", "Research Paper"),
        session_id=session_id,
        analysis_markdown=analysis,
        final_markdown=final_draft,
        out_path=out_path,
        resource_dir=tmp_dir,
    )
    return FileResponse(
        path=str(out_path),
        filename=f"delve-paper-{session_id[:8]}.pdf",
        media_type="application/pdf",
    )


@router.get("/{session_id}/slides")
async def get_presentation_slides(
    session_id: str, user: AuthenticatedUser = Depends(get_current_user),
) -> None:
    await _get_owned_session(session_id, user)
    raise HTTPException(status_code=410, detail="Slides are temporarily disabled for security hardening")


@router.get("/sessions/list")
async def list_sessions(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, list[dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    for data in await supabase_repository.list_sessions(user.id):
        sid = str(data["id"])
        sessions.append({
            "session_id": sid,
            "topic": data.get("topic", ""),
            "status": data.get("status", "unknown"),
            "current_step": data.get("current_step", "unknown"),
            "created_at": data.get("started_at", ""),
            "updated_at": data.get("updated_at", ""),
            "token_estimate": data.get("metrics", {}).get("token_estimate", 0),
            "cost_estimate_usd": data.get("metrics", {}).get("cost_estimate_usd", 0),
        })
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"sessions": sessions}


@router.get("/quota")
async def get_quota(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    """Get user's paper generation quota information."""
    quota = await supabase_repository.get_user_quota_info(user.id)
    remaining = quota["free_papers_allowed"] - quota["papers_generated"]
    return {
        "papers_generated": quota["papers_generated"],
        "papers_allowed": quota["free_papers_allowed"],
        "papers_remaining": max(0, remaining),
        "last_paper_at": quota["last_paper_at"],
        "has_quota": remaining > 0,
    }

