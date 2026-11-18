"""
Chat route — POST /chat

Accepts a question (optionally with a conversation_id) and returns an
answer with citations from the uploaded documents.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    RetrievedChunk,
    RetrievalDebug,
    Source,
)
from app.services import rag_engine

router = APIRouter(tags=["chat"])

# ── In-memory conversation store ────────────────────────────────────────────
# Maps conversation_id → list of {"role": ..., "content": ...}. Capped with
# LRU eviction so long-running servers do not leak memory.
_conversations: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()

MAX_HISTORY_MESSAGES = 20  # last 10 question/answer pairs


def _store_history(
    conversation_id: str, history: list[dict[str, str]]
) -> None:
    """Save history and evict the least-recently-used conversations."""
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    _conversations[conversation_id] = history
    _conversations.move_to_end(conversation_id)
    max_conversations = max(settings.MAX_CONVERSATIONS, 1)
    while len(_conversations) > max_conversations:
        _conversations.popitem(last=False)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a user question and return an answer with citations.

    Declared as a sync endpoint so FastAPI runs the blocking LLM call in its
    threadpool instead of blocking the event loop.

    If ``conversation_id`` is omitted a new one is generated.
    History is capped at the last 20 messages (10 question/answer pairs).
    """
    # Resolve or create conversation ID
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Fetch existing history
    history = _conversations.get(conversation_id, [])

    try:
        result = rag_engine.generate_answer(
            question=request.question,
            conversation_history=history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Update history
    updated_history = history + [
        {"role": "user", "content": request.question},
        {"role": "assistant", "content": result["answer"]},
    ]
    _store_history(conversation_id, updated_history)

    # Build response
    sources = [
        Source(document=s["document"], page=s["page"], text=s["text"])
        for s in result.get("sources", [])
    ]

    debug_data = result.get("debug")
    debug = (
        RetrievalDebug(
            original_question=debug_data["original_question"],
            rewritten_question=debug_data["rewritten_question"],
            chunks=[RetrievedChunk(**chunk) for chunk in debug_data.get("chunks", [])],
        )
        if debug_data
        else None
    )

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        conversation_id=conversation_id,
        has_answer=result["has_answer"],
        debug=debug,
    )
