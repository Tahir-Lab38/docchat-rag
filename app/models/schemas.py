"""
Pydantic models for all API request / response payloads.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Chat ────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat question."""

    question: str = Field(..., min_length=1, description="User question text")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to continue a session",
    )


class Source(BaseModel):
    """A single citation source."""

    document: str = Field(..., description="Source document name")
    page: int = Field(..., description="Page number in the document")
    text: str = Field(..., description="Relevant text snippet from the source")


class RetrievedChunk(BaseModel):
    """A chunk used during retrieval, for the debug view."""

    document: str
    page: int
    score: float
    text: str


class RetrievalDebug(BaseModel):
    """Retrieval diagnostics shown in the sources panel."""

    original_question: str
    rewritten_question: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response to a chat question."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    conversation_id: str
    has_answer: bool
    debug: RetrievalDebug | None = None


# ── Documents ───────────────────────────────────────────────────────────────


class DocumentResponse(BaseModel):
    """Metadata for an uploaded document."""

    id: str
    name: str
    pages: int
    chunks: int
    uploaded_at: str


class UploadResponse(BaseModel):
    """Response after a successful document upload."""

    message: str
    document: DocumentResponse


# ── Evaluation ──────────────────────────────────────────────────────────────


class EvalResult(BaseModel):
    """Result of evaluating a single question."""

    question: str
    expected: str
    actual: str
    score: float
    sources_found: bool
