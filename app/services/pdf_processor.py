"""
PDF text extraction and chunking service.

Uses PyMuPDF (fitz) for reliable, fast PDF text extraction and
implements sentence-aware overlapping chunking for better retrieval quality.
"""

from __future__ import annotations

import os
import re
from typing import Any

import fitz  # PyMuPDF

from app.config import settings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z0-9])")
_MIN_CHUNK_CHARS = 40


def extract_text_from_pdf(file_path: str) -> list[dict[str, Any]]:
    """Extract text from each page of a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts ``[{"page": 1, "text": "..."}, ...]`` (1-indexed pages).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        RuntimeError: If the PDF cannot be opened.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages: list[dict[str, Any]] = []
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    try:
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            # Skip truly empty pages
            if text:
                pages.append({"page": page_num + 1, "text": text})
    finally:
        doc.close()
    return pages


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences without cutting abbreviations mid-token."""
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _split_long_sentence(text: str, size: int) -> list[str]:
    """Hard-split an oversized sentence on word boundaries."""
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in text.split():
        added = len(word) + (1 if current else 0)
        if current and current_len + added > size:
            parts.append(" ".join(current))
            current, current_len = [], 0
            added = len(word)
        current.append(word)
        current_len += added
    if current:
        parts.append(" ".join(current))
    return parts


def _overlap_prefix(chunk: str, overlap: int) -> str:
    """Return the tail of a chunk to carry into the next one."""
    if overlap <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(_split_sentences(chunk)):
        added = len(sentence) + (1 if tail else 0)
        if tail and total + added > overlap:
            break
        tail.insert(0, sentence)
        total += added
    if tail:
        return " ".join(tail)
    return chunk[-overlap:].lstrip()


def chunk_text(
    pages: list[dict[str, Any]],
    chunk_size: int | None = None,
    overlap: int | None = None,
    document_name: str = "unknown",
) -> list[dict[str, Any]]:
    """Split page texts into sentence-aware overlapping chunks with metadata.

    Sentences are never split unless a single sentence is longer than
    ``chunk_size``.  Consecutive chunks share an overlap so context is not
    lost at boundaries.

    Args:
        pages: Output of :func:`extract_text_from_pdf`.
        chunk_size: Maximum characters per chunk (default from settings).
        overlap: Number of overlapping characters between consecutive chunks.
        document_name: Name of the source document (used in metadata).

    Returns:
        List of chunk dicts with ``text``, ``page``, ``chunk_index`` and
        ``document_name`` keys.
    """
    chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    overlap = min(overlap, max(chunk_size // 2, 0))

    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for page_data in pages:
        page_num: int = page_data["page"]
        text: str = re.sub(r"[ \t]+", " ", page_data["text"]).strip()
        if len(text) < _MIN_CHUNK_CHARS:
            continue

        buffer = ""

        def emit(chunk_body: str) -> None:
            nonlocal chunk_index
            body = chunk_body.strip()
            if len(body) >= _MIN_CHUNK_CHARS:
                chunks.append(
                    {
                        "text": body,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "document_name": document_name,
                    }
                )
                chunk_index += 1

        for sentence in _split_sentences(text):
            pieces = (
                _split_long_sentence(sentence, chunk_size)
                if len(sentence) > chunk_size
                else [sentence]
            )
            for piece in pieces:
                candidate = f"{buffer} {piece}".strip() if buffer else piece
                if buffer and len(candidate) > chunk_size:
                    emit(buffer)
                    buffer = f"{_overlap_prefix(buffer, overlap)} {piece}".strip()
                else:
                    buffer = candidate

                while len(buffer) > chunk_size:
                    cutoff = max(chunk_size - overlap, 1)
                    emit(buffer[:chunk_size])
                    buffer = buffer[cutoff:].lstrip()

        if buffer:
            emit(buffer)

    return chunks
