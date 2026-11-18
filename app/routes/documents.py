"""
Document management routes.

- POST   /documents/upload   — upload a PDF, extract text, chunk, embed
- GET    /documents          — list all uploaded documents
- DELETE /documents/{doc_id} — remove a document and its vectors
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.config import settings
from app.models.schemas import DocumentResponse, UploadResponse
from app.services import pdf_processor, vector_store

router = APIRouter(prefix="/documents", tags=["documents"])

# ── Document registry ───────────────────────────────────────────────────────
# Loaded once from the vector store on first access so uploads survive
# restarts without scanning every embedding on each request.
_documents: dict[str, dict[str, Any]] = {}
_registry_loaded = False


def _ensure_documents_loaded() -> None:
    """Load the registry from the persistent vector store exactly once."""
    global _registry_loaded
    if _registry_loaded:
        return
    for item in vector_store.get_document_metadata():
        _documents.setdefault(item["id"], item)
    _registry_loaded = True


def _response(doc: dict[str, Any]) -> DocumentResponse:
    return DocumentResponse(
        id=doc["id"],
        name=doc["name"],
        pages=doc["pages"],
        chunks=doc["chunks"],
        uploaded_at=doc["uploaded_at"],
    )


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF document, extract text, chunk it, and embed into ChromaDB.

    Files with identical content are only indexed once. This is a sync
    endpoint so the CPU-heavy PDF processing runs in FastAPI's threadpool.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    contents = file.file.read()

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {settings.MAX_UPLOAD_MB} MB.",
        )
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    file_hash = hashlib.sha256(contents).hexdigest()

    _ensure_documents_loaded()

    existing_id = vector_store.find_document_by_hash(file_hash)
    if existing_id and existing_id in _documents:
        return UploadResponse(
            message=f"Document '{_documents[existing_id]['name']}' was already uploaded.",
            document=_response(_documents[existing_id]),
        )

    doc_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename) or "document.pdf"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{safe_name}")

    try:
        with open(file_path, "wb") as fh:
            fh.write(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {exc}",
        ) from exc

    try:
        pages = pdf_processor.extract_text_from_pdf(file_path)
        chunks = pdf_processor.chunk_text(
            pages,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
            document_name=safe_name,
        )
        if not chunks:
            raise ValueError("No readable text found in the PDF.")
    except Exception as exc:
        if os.path.isfile(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {exc}",
        ) from exc

    uploaded_at = datetime.now(timezone.utc).isoformat()
    try:
        vector_store.add_document(
            doc_id,
            chunks,
            file_hash=file_hash,
            num_pages=len(pages),
            uploaded_at=uploaded_at,
            file_path=file_path,
        )
    except Exception as exc:
        if os.path.isfile(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed document: {exc}",
        ) from exc

    doc_meta = {
        "id": doc_id,
        "name": safe_name,
        "pages": len(pages),
        "chunks": len(chunks),
        "uploaded_at": uploaded_at,
        "file_path": file_path,
        "file_hash": file_hash,
    }
    _documents[doc_id] = doc_meta

    return UploadResponse(
        message=f"Document '{safe_name}' uploaded and processed successfully.",
        document=_response(doc_meta),
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents() -> list[DocumentResponse]:
    """Return metadata for all uploaded documents."""
    _ensure_documents_loaded()
    return [_response(doc) for doc in _documents.values()]


@router.delete("/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str]:
    """Delete a document by ID — removes vectors and the uploaded file."""
    _ensure_documents_loaded()
    if doc_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        vector_store.delete_document(doc_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove vectors: {exc}",
        ) from exc

    file_path = _documents[doc_id].get("file_path", "")
    if file_path and os.path.isfile(file_path):
        os.remove(file_path)

    del _documents[doc_id]
    return {"message": f"Document '{doc_id}' deleted successfully."}
