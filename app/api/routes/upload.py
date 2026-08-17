"""
Delve PDF Upload Route
───────────────────────
POST /upload/pdf – Upload a PDF, extract text, chunk, embed, and store in Supabase pgvector.
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from pydantic import BaseModel

from app.core.auth import AuthenticatedUser, get_current_user
from app.core.config import settings
from app.core.rate_limit import enforce_rate_limit
from app.core.supabase import supabase_repository
from app.services.parsing import inspect_pdf, chunk_text
from app.services.vector_store import vector_store

logger = logging.getLogger("delve.api.upload")

router = APIRouter(prefix="/upload", tags=["upload"])
READ_CHUNK_BYTES = 1024 * 1024


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    chunks_stored: int
    total_characters: int


@router.post("/pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Upload a PDF file.

    Extracts text, splits into overlapping chunks, and stores
    in ChromaDB for later retrieval during research.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    await enforce_rate_limit(
        key=f"upload:{user.id}", limit=10, window_seconds=3600,
        message="Hourly upload limit reached",
    )
    if await supabase_repository.count_documents(user.id) >= settings.max_documents_per_user:
        raise HTTPException(status_code=429, detail="Document limit reached; delete an existing document first")

    contents = bytearray()
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        if len(contents) + len(chunk) > settings.max_upload_bytes:
            raise HTTPException(status_code=400, detail="File too large for this deployment")
        contents.extend(chunk)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    if not bytes(contents[:5]).startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File does not look like a PDF")

    # The document id is generated server-side and is never shared across users.
    file_id = str(uuid.uuid4())
    storage_path = f"{user.id}/uploads/{file_id}.pdf"

    try:
        await supabase_repository.put_document(
            path=storage_path, content=bytes(contents), content_type="application/pdf"
        )
        await supabase_repository.create_document({
            "id": file_id,
            "owner_id": user.id,
            "filename": file.filename,
            "storage_path": storage_path,
            "size_bytes": len(contents),
            "status": "processing",
        })

        page_count, text = await asyncio.to_thread(
            inspect_pdf,
            bytes(contents),
            max_pages=settings.max_upload_pages,
            max_characters=settings.max_extracted_text_bytes,
        )

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Chunk the text
        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated from PDF")

        embeddings = await asyncio.to_thread(vector_store.embed, chunks)
        metadata = {
            "filename": file.filename,
            "file_size": len(contents),
            "page_count": page_count,
        }
        await supabase_repository.insert_document_chunks([
            {
                "document_id": file_id,
                "owner_id": user.id,
                "chunk_index": index,
                "content": chunk,
                "metadata": metadata,
                "embedding": embedding,
            }
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ])
        chunks_stored = len(chunks)
        await supabase_repository.update_document(file_id, user.id, {
            "status": "ready", "page_count": page_count, "extracted_characters": len(text),
        })

        logger.info("Uploaded PDF '%s' -> %d chunks stored (id: %s)", file.filename, chunks_stored, file_id)

        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            chunks_stored=chunks_stored,
            total_characters=len(text),
        )

    except HTTPException:
        try:
            await supabase_repository.update_document(file_id, user.id, {"status": "error", "error": "PDF validation failed"})
        except Exception:
            pass
        raise
    except ValueError as e:
        try:
            await supabase_repository.update_document(file_id, user.id, {"status": "error", "error": str(e)[:300]})
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        try:
            await supabase_repository.update_document(file_id, user.id, {"status": "error", "error": str(e)[:300]})
        except Exception:
            pass
        logger.error("PDF upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process uploaded PDF")
