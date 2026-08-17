"""Hosted pgvector retrieval for user-owned uploaded PDFs."""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.supabase import supabase_repository
from app.services.vector_store import vector_store


async def query_uploaded_documents(
    *, owner_id: str, document_ids: list[str], query: str, n_results: int = 5
) -> list[dict[str, Any]]:
    embedding = (await asyncio.to_thread(vector_store.embed, [query]))[0]
    rows = await supabase_repository.match_document_chunks(
        owner_id=owner_id, document_ids=document_ids, embedding=embedding, n_results=n_results,
    )
    return [
        {"text": row.get("content", ""), "metadata": row.get("metadata") or {}, "distance": row.get("distance")}
        for row in rows
    ]
