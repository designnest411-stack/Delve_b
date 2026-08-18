"""Small async Supabase REST/Storage adapter used by the FastAPI service."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote
from datetime import datetime, timezone

import httpx

from app.core.config import settings

logger = logging.getLogger("delve.supabase")


class SupabaseRepository:
    """Server-side-only data access. The service-role key never reaches clients."""

    def _require_configured(self) -> None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise RuntimeError("Supabase server configuration is missing")

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        self._require_configured()
        return {
            "apikey": settings.supabase_service_role_key,
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
            **(extra or {}),
        }

    async def _rest(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        payload: Any | None = None,
        prefer: str = "return=representation",
    ) -> Any:
        headers = self._headers({"Content-Type": "application/json", "Prefer": prefer})
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                f"{settings.supabase_rest_url}/{table}",
                params=params,
                headers=headers,
                json=payload,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"Supabase {table} request failed ({response.status_code}): {response.text[:300]}")
        if not response.content:
            return []
        result = response.json()
        return result if result is not None else []

    @staticmethod
    def _session_record(session_id: str, session: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "id": session_id,
            "owner_id": session["owner_id"],
            "topic": session.get("topic", "") or "Research session",
            "status": session.get("status", "initializing"),
            "current_step": session.get("current_step", "initializing"),
            "controls": session.get("controls") if isinstance(session.get("controls"), dict) else {},
            "uploaded_paper_ids": session.get("uploaded_paper_ids") if isinstance(session.get("uploaded_paper_ids"), list) else [],
            "timeline": session.get("timeline") if isinstance(session.get("timeline"), list) else [],
            "result": session.get("result") if isinstance(session.get("result"), dict) else {},
            "metrics": session.get("metrics") if isinstance(session.get("metrics"), dict) else {},
            "chat_history": session.get("chat_history") if isinstance(session.get("chat_history"), list) else [],
            "resource_files": session.get("resource_files") if isinstance(session.get("resource_files"), list) else [],
            "error": session.get("error"),
            "started_at": session.get("started_at"),
            "finished_at": session.get("finished_at"),
            "updated_at": session.get("updated_at") or now,
        }

    @staticmethod
    def session_from_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record.get("id"),
            "owner_id": record["owner_id"],
            "topic": record.get("topic", ""),
            "status": record.get("status", "unknown"),
            "current_step": record.get("current_step", "unknown"),
            "controls": record.get("controls") or {},
            "uploaded_paper_ids": record.get("uploaded_paper_ids") or [],
            "timeline": record.get("timeline") or [],
            "result": record.get("result") or {},
            "metrics": record.get("metrics") or {},
            "chat_history": record.get("chat_history") or [],
            "resource_files": record.get("resource_files") or [],
            "error": record.get("error"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "updated_at": record.get("updated_at"),
        }

    async def upsert_session(self, session_id: str, session: dict[str, Any]) -> None:
        await self._rest(
            "POST",
            "research_sessions",
            params={"on_conflict": "id"},
            payload=self._session_record(session_id, session),
            prefer="resolution=merge-duplicates,return=minimal",
        )

    async def get_session(self, session_id: str, owner_id: str) -> dict[str, Any] | None:
        rows = await self._rest(
            "GET", "research_sessions",
            params={"id": f"eq.{session_id}", "owner_id": f"eq.{owner_id}", "select": "*"},
        )
        return self.session_from_record(rows[0]) if rows else None

    async def list_sessions(self, owner_id: str) -> list[dict[str, Any]]:
        return await self._rest(
            "GET", "research_sessions",
            params={
                "owner_id": f"eq.{owner_id}",
                "select": "id,owner_id,topic,status,current_step,metrics,started_at,updated_at,finished_at,error",
                "order": "updated_at.desc",
            },
        )

    async def delete_session(self, session_id: str, owner_id: str) -> None:
        await self._rest(
            "DELETE", "research_sessions",
            params={"id": f"eq.{session_id}", "owner_id": f"eq.{owner_id}"},
            prefer="return=minimal",
        )

    async def cancel_orphaned_jobs(self, session_id: str) -> None:
        """Cancel any queued/running jobs for a session (used before retry or cancel)."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            await self._rest(
                "PATCH", "research_jobs",
                params={"session_id": f"eq.{session_id}", "status": "in.(queued,running)"},
                payload={"status": "cancelled", "error": "Superseded or cancelled", "completed_at": now, "updated_at": now},
                prefer="return=minimal",
            )
        except Exception as exc:
            logger.warning("Could not cancel orphaned jobs for session %s: %s", session_id, exc)

    async def create_job(self, session_id: str, owner_id: str, kind: str = "research") -> dict[str, Any]:
        rows = await self._rest(
            "POST", "research_jobs",
            payload={"session_id": session_id, "owner_id": owner_id, "kind": kind, "status": "queued"},
        )
        return rows[0]

    async def count_active_jobs(self, owner_id: str) -> int:
        rows = await self._rest(
            "GET", "research_jobs",
            params={"owner_id": f"eq.{owner_id}", "status": "in.(queued,running)", "select": "id"},
        )
        return len(rows) if isinstance(rows, list) else 0

    async def check_user_quota(self, user_id: str) -> bool:
        """Check if user has remaining quota (5 free papers lifetime)."""
        try:
            rows = await self._rest("POST", "rpc/check_user_quota", payload={"p_user_id": user_id})
            return rows if isinstance(rows, bool) else rows[0] if rows else True
        except Exception as exc:
            logger.warning("check_user_quota failed, defaulting to allowed: %s", exc)
            return True

    async def get_user_quota_info(self, user_id: str) -> dict[str, Any]:
        """Get detailed quota information for display."""
        try:
            rows = await self._rest(
                "GET", "user_paper_quotas",
                params={"user_id": f"eq.{user_id}", "select": "papers_generated,free_papers_allowed,last_paper_at"},
            )
            if not rows:
                return {"papers_generated": 0, "free_papers_allowed": 5, "last_paper_at": None}
            return rows[0]
        except Exception as exc:
            logger.warning("get_user_quota_info failed, returning default quota: %s", exc)
            return {"papers_generated": 0, "free_papers_allowed": 5, "last_paper_at": None}

    async def claim_job(self, job_id: str) -> dict[str, Any] | None:
        rows = await self._rest("POST", "rpc/claim_research_job", payload={"p_job_id": job_id})
        return rows[0] if rows else None

    async def finish_job(self, job_id: str, status: str, error: str | None = None) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        # Guard: never overwrite a completed job with an error (handles QStash retry races).
        params: dict[str, str] = {"id": f"eq.{job_id}"}
        if status != "complete":
            params["status"] = "neq.complete"
        await self._rest(
            "PATCH", "research_jobs",
            params=params,
            payload={"status": status, "error": error, "completed_at": now, "updated_at": now},
            prefer="return=minimal",
        )

    async def insert_llm_usage(self, usage: dict[str, Any]) -> None:
        """Persist the measured usage returned by the LLM provider."""
        await self._rest("POST", "llm_usage", payload=usage, prefer="return=minimal")

    async def put_document(self, *, path: str, content: bytes, content_type: str) -> None:
        self._require_configured()
        safe_path = quote(path, safe="/")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.supabase_storage_url}/object/{settings.supabase_storage_bucket}/{safe_path}",
                headers=self._headers({"Content-Type": content_type, "x-upsert": "true"}),
                content=content,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"Supabase Storage upload failed ({response.status_code}): {response.text[:300]}")

    async def create_document(self, document: dict[str, Any]) -> dict[str, Any]:
        rows = await self._rest("POST", "research_documents", payload=document)
        return rows[0]

    async def insert_document_chunks(self, chunks: list[dict[str, Any]]) -> None:
        # PostgREST accepts batch inserts; bounded chunks keep payloads friendly
        # to Supabase and avoid a huge single request for a long PDF.
        for start in range(0, len(chunks), 100):
            await self._rest(
                "POST", "document_chunks", payload=chunks[start:start + 100], prefer="return=minimal"
            )

    async def update_document(self, document_id: str, owner_id: str, values: dict[str, Any]) -> None:
        await self._rest(
            "PATCH", "research_documents",
            params={"id": f"eq.{document_id}", "owner_id": f"eq.{owner_id}"},
            payload=values, prefer="return=minimal",
        )

    async def count_documents(self, owner_id: str) -> int:
        rows = await self._rest(
            "GET", "research_documents",
            params={"owner_id": f"eq.{owner_id}", "status": "neq.deleted", "select": "id"},
        )
        return len(rows)

    async def get_documents(self, document_ids: list[str], owner_id: str) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        return await self._rest(
            "GET", "research_documents",
            params={"id": f"in.({','.join(document_ids)})", "owner_id": f"eq.{owner_id}", "select": "*"},
        )

    async def match_document_chunks(
        self, *, owner_id: str, document_ids: list[str], embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        vector = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        return await self._rest(
            "POST", "rpc/match_document_chunks",
            payload={
                "p_owner_id": owner_id,
                "p_document_ids": document_ids,
                "p_query_embedding": vector,
                "p_match_count": n_results,
            },
        )


supabase_repository = SupabaseRepository()
