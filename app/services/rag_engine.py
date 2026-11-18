"""
RAG (Retrieval-Augmented Generation) engine.

Orchestrates the full pipeline: rewrite follow-up → retrieve relevant chunks
→ build prompt → call Groq LLM → format response with citations.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from groq import Groq

from app.config import settings
from app.prompts.templates import (
    QUESTION_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.services import vector_store

logger = logging.getLogger(__name__)

# ── Groq client (lazy singleton) ────────────────────────────────────────────

_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    """Return a Groq client, creating it on first call."""
    global _groq_client
    if _groq_client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Add it to your .env file."
            )
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a labelled context block for the prompt."""
    if not chunks:
        return "(No relevant context found.)"

    parts: list[str] = []
    for chunk in chunks:
        label = f"[Source: {chunk['document_name']}, Page {chunk['page_number']}]"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n".join(parts)


def _build_history_block(
    history: list[dict[str, str]], limit: int = 6
) -> str:
    """Render recent conversation history as a readable block."""
    if not history:
        return "(No prior conversation.)"
    lines: list[str] = []
    for msg in history[-limit:]:
        role = msg.get("role", "user").capitalize()
        lines.append(f"{role}: {msg.get('content', '')[:400]}")
    return "\n".join(lines)


def _extract_sources(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert chunk metadata into the source list for the response."""
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk["document_name"], chunk["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "document": chunk["document_name"],
                    "page": chunk["page_number"],
                    "text": chunk["text"][:300],
                }
            )
    return sources


_NO_ANSWER_PHRASES = [
    "i don't have enough information",
    "i do not have enough information",
    "not enough information in the provided documents",
    "cannot answer this question",
    "no relevant context",
]

_REFUSAL = (
    "I don't have enough information in the provided documents to answer this question."
)

_QUESTION_REQUEST = re.compile(
    r"\b(generate|create|make|write|give|produce)\b.*\bquestions?\b",
    re.IGNORECASE,
)


def _check_has_answer(answer: str) -> bool:
    """Detect whether the LLM indicated it could not answer."""
    lower = answer.lower()
    return not any(phrase in lower for phrase in _NO_ANSWER_PHRASES)


def _create_completion(
    client: Groq,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Groq once, retrying only on rate limits, and return plain content."""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()  # type: ignore[union-attr]
        except Exception as exc:
            if attempt == 0 and "rate_limit" in str(exc).lower():
                logger.warning("Groq rate limit hit; retrying once: %s", exc)
                time.sleep(3)
                continue
            raise
    return ""


def _rewrite_query(
    question: str,
    history: list[dict[str, str]],
    client: Groq,
) -> str:
    """Turn a follow-up into a standalone search query using conversation history."""
    if not history:
        return question

    prompt = QUERY_REWRITE_PROMPT.format(
        history=_build_history_block(history),
        question=question,
    )
    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for model in (settings.QUERY_REWRITE_MODEL, settings.QUERY_REWRITE_FALLBACK_MODEL):
        try:
            rewritten = _create_completion(client, model, messages, 128, 0.0)
            rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
            if rewritten:
                if model != settings.QUERY_REWRITE_MODEL:
                    logger.info("Query rewrite used fallback model '%s'", model)
                return rewritten
        except Exception as exc:
            logger.warning(
                "Query rewrite model '%s' failed (%s); trying fallback", model, exc
            )

    logger.warning("Query rewrite unavailable; using original question")
    return question


def _log_retrieval(
    original: str, rewritten: str, chunks: list[dict[str, Any]]
) -> None:
    """Log the retrieval inputs and every chunk for debugging."""
    logger.info("Retrieval original question: %s", original)
    logger.info("Retrieval rewritten question: %s", rewritten)
    logger.info("Retrieved %d chunks:", len(chunks))
    for i, chunk in enumerate(chunks, start=1):
        distance = float(chunk.get("distance", 1.0))
        logger.info(
            "  [%d] page=%s score=%.3f distance=%.3f %s",
            i,
            chunk["page_number"],
            max(0.0, 1.0 - distance),
            distance,
            chunk["text"][:150].replace("\n", " "),
        )


# ── Public API ──────────────────────────────────────────────────────────────


def generate_answer(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the full RAG pipeline for a user question.

    Args:
        question: The user's original natural-language question.
        conversation_history: Optional prior ``{"role": ..., "content": ...}`` dicts.

    Returns:
        Dict with keys ``answer``, ``sources``, ``has_answer`` and ``debug``.
    """
    conversation_history = conversation_history or []
    client = get_groq_client()

    # 1. Rewrite follow-ups, then retrieve using the standalone query only
    search_query = _rewrite_query(question, conversation_history, client)
    chunks = vector_store.query(search_query, top_k=settings.TOP_K)
    _log_retrieval(question, search_query, chunks)

    debug: dict[str, Any] = {
        "original_question": question,
        "rewritten_question": search_query,
        "chunks": [
            {
                "document": chunk["document_name"],
                "page": chunk["page_number"],
                "score": round(max(0.0, 1.0 - float(chunk.get("distance", 1.0))), 3),
                "text": chunk["text"][:150],
            }
            for chunk in chunks
        ],
    }

    # Refuse only when nothing was retrieved at all
    if not chunks:
        return {"answer": _REFUSAL, "sources": [], "has_answer": False, "debug": debug}

    # 2. Build the prompt — always send the ORIGINAL question
    context_block = _build_context_block(chunks)

    is_generation = bool(_QUESTION_REQUEST.search(question))
    template = QUESTION_GENERATION_PROMPT if is_generation else RAG_PROMPT_TEMPLATE
    user_prompt = template.format(context=context_block, question=question)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for msg in conversation_history[-20:]:
        role = msg.get("role")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_prompt})

    # 3. Call Groq LLM. gpt-oss models sometimes put the answer in `reasoning`
    # and leave `content` empty, so retry with the fallback model in that case.
    max_tokens = 2048 if is_generation else settings.LLM_MAX_TOKENS
    try:
        answer = _create_completion(
            client, settings.GROQ_MODEL, messages, max_tokens, settings.LLM_TEMPERATURE
        )
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc)
        return {
            "answer": "The answer service is busy right now. Please try again in a moment.",
            "sources": [],
            "has_answer": False,
            "debug": debug,
        }

    if not answer:
        logger.warning("Primary model returned empty content; using fallback model")
        try:
            answer = _create_completion(
                client,
                settings.ANSWER_FALLBACK_MODEL,
                messages,
                max_tokens,
                settings.LLM_TEMPERATURE,
            )
        except Exception as exc:
            logger.warning("Fallback answer call failed: %s", exc)

    # A refusal is only valid for zero retrieved chunks. Retry once with an
    # explicit instruction when it happens despite relevant excerpts.
    if not _check_has_answer(answer) and not is_generation:
        logger.warning("Model refused despite %d retrieved chunks; retrying", len(chunks))
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Use the excerpts above. If they contain only part of the "
                    "answer, give that part and state what is missing. Do not "
                    "say you don't have enough information."
                ),
            }
        ]
        for model in (settings.GROQ_MODEL, settings.ANSWER_FALLBACK_MODEL):
            try:
                candidate = _create_completion(
                    client, model, retry_messages, settings.LLM_MAX_TOKENS,
                    settings.LLM_TEMPERATURE,
                )
                if candidate:
                    answer = candidate
                    break
            except Exception as exc:
                logger.warning("Refusal recovery with '%s' failed: %s", model, exc)

    if not answer:
        logger.warning("No model produced answer text for: %s", question[:120])
        answer = (
            "Excerpts were found, but no answer text could be generated. "
            "Please rephrase the question."
        )

    has_answer = _check_has_answer(answer)
    sources = _extract_sources(chunks) if has_answer else []

    logger.info(
        "Answer generated (%d chars, has_answer=%s) for: %s",
        len(answer),
        has_answer,
        question[:120],
    )
    return {
        "answer": answer.strip(),
        "sources": sources,
        "has_answer": has_answer,
        "debug": debug,
    }
