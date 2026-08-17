"""
Delve – Multi-Agent Deep Research System
═════════════════════════════════════════
Main FastAPI application with WebSocket support.

Endpoints:
  POST   /research/start          – Start a new research session
  GET    /research/{id}/status    – Get session status
  GET    /research/{id}/paper     – Get completed paper
  GET    /research/sessions/list  – List all sessions
  POST   /upload/pdf              – Upload a PDF for inclusion
  WS     /ws/{session_id}         – Live progress updates
  GET    /health                  – Health check
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.auth import verify_ws_ticket
from app.core.supabase import supabase_repository
from app.core.llm_client import llm_client
from app.api.routes.research import (
    router as research_router,
    ws_connections,
    flush_pending_persists,
)
from app.api.routes.upload import router as upload_router

# ── Logging Setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("delve.main")


def _validate_configuration() -> None:
    """Refuse to boot with missing security boundaries (production/staging only)."""
    if not settings.is_production:
        logger.info("APP_ENVIRONMENT=%s — skipping production configuration check", settings.app_environment)
        return
    required = {
        "SUPABASE_URL": settings.supabase_url,
        "SUPABASE_SERVICE_ROLE_KEY": settings.supabase_service_role_key,
        "WS_TICKET_SECRET": settings.ws_ticket_secret,
        "JOB_DISPATCH_SECRET": settings.job_dispatch_secret,
        "QSTASH_TOKEN": settings.qstash_token,
        "PUBLIC_API_BASE_URL": settings.public_api_base_url,
        "UPSTASH_REDIS_REST_URL": settings.upstash_redis_rest_url,
        "UPSTASH_REDIS_REST_TOKEN": settings.upstash_redis_rest_token,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Deployment configuration is incomplete: {', '.join(missing)}")


# ── Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Delve – Multi-Agent Deep Research System")
    logger.info("=" * 60)
    _validate_configuration()

    async def _validate_llm_in_background() -> None:
        logger.info("Validating LLM API key in background...")
        valid = await llm_client.validate_api_key()
        if valid:
            logger.info("✓ LLM API key is valid")
        else:
            logger.warning("✗ LLM API key validation failed – LLM calls may fail")

    validation_task = asyncio.create_task(_validate_llm_in_background())

    yield

    # Shutdown
    if not validation_task.done():
        validation_task.cancel()
    await flush_pending_persists()
    await llm_client.close()
    logger.info("Delve shut down cleanly")


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Delve API",
    description="Multi-Agent Deep Research & Paper Drafting System",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_origin_regex=r"^https://([a-zA-Z0-9_-]+\.)*vercel\.app$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Delve-Job-Secret"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'"
    )
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── Include Routers ───────────────────────────────────────────────────────
app.include_router(research_router)
app.include_router(upload_router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Delve — Multi-Agent Deep Research System",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ── WebSocket Endpoint ────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for live research progress updates.

    Clients must include a short-lived ticket obtained from POST /research/{id}/ws-ticket.
    Messages sent by the server:
      {"type": "status",      "message": "..."}
      {"type": "paper_found", "data": {...}}
      {"type": "gap",         "data": {...}}
      {"type": "complete",    "message": "...", "data": {...}}
      {"type": "error",       "message": "..."}
      {"type": "heartbeat"}
    """
    ticket = websocket.query_params.get("ticket", "")
    try:
        user = verify_ws_ticket(ticket, session_id=session_id)
        session = await supabase_repository.get_session(session_id, user.id)
        if not session:
            await websocket.close(code=4403)
            return
    except Exception:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    if session_id not in ws_connections:
        ws_connections[session_id] = []
    ws_connections[session_id].append(websocket)

    try:
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected to session {session_id}",
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "stop":
                        logger.info("Client requested stop for session %s", session_id)
                        await websocket.send_json({
                            "type": "status",
                            "message": "Stop requested – job will complete current step before halting.",
                        })
                except Exception:
                    pass
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    except Exception as e:
        logger.error("WebSocket error for session %s: %s", session_id, e)
    finally:
        if session_id in ws_connections and websocket in ws_connections[session_id]:
            ws_connections[session_id].remove(websocket)


# ── Health Check ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Delve",
        "version": "1.0.0",
    }


# ── Main Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )
