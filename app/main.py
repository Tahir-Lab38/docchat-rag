"""
DocChat — FastAPI application entry-point.

Sets up the app with:
- Lifespan events (initialise ChromaDB on startup)
- CORS middleware (permissive for development)
- All route routers
- Static file serving for the frontend
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import chat, documents, health
from app.services.vector_store import init_vector_store

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise resources on startup."""
    logger.info("Starting DocChat — initialising vector store …")

    if not settings.GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY is not set. Chat requests will fail until it is configured."
        )

    # Ensure storage directories exist
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    init_vector_store()
    logger.info("Vector store ready.")

    yield  # application runs here

    logger.info("Shutting down DocChat.")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocChat API",
    description="AI-powered document Q&A assistant with RAG, citations, and hallucination prevention.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────

_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)

# ── Static frontend ────────────────────────────────────────────────────────

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
