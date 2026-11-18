"""
ChromaDB vector store wrapper.

Provides a clean interface over ChromaDB for adding, querying, and
deleting document embeddings.  Uses SentenceTransformer embeddings.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

# Disable Chroma telemetry and progress bars before importing Chroma.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TQDM_DISABLE", "1")

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings

logger = logging.getLogger(__name__)

# Some Chroma versions still emit telemetry errors even when disabled.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# ── Module-level singleton ──────────────────────────────────────────────────

_client: chromadb.ClientAPI | None = None
_embedding_fn: SentenceTransformerEmbeddingFunction | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "docchat_documents"


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    """Lazily create the embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL,
        )
    return _embedding_fn


def init_vector_store() -> None:
    """Initialise (or re-open) the ChromaDB persistent client and collection.

    Safe to call multiple times; the collection is only created once per
    process.
    """
    global _client, _collection

    if _collection is not None:
        return

    _client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "Vector store initialised — collection '%s' has %d items",
        COLLECTION_NAME,
        _collection.count(),
    )


def count() -> int:
    """Return the number of embeddings currently stored."""
    return _get_collection().count()


def _get_collection() -> chromadb.Collection:
    """Return the active collection, initialising it lazily if needed."""
    if _collection is None:
        init_vector_store()
    if _collection is None:
        raise RuntimeError(
            "Vector store not initialised. Call init_vector_store() first."
        )
    return _collection


def _text_key(text: str) -> str:
    """Normalise text so identical chunks from duplicate uploads collapse."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


# ── Public API ──────────────────────────────────────────────────────────────


def add_document(
    doc_id: str,
    chunks: list[dict[str, Any]],
    *,
    file_hash: str = "",
    num_pages: int = 0,
    uploaded_at: str = "",
    file_path: str = "",
) -> int:
    """Add chunked document embeddings to the vector store.

    Args:
        doc_id: Unique document identifier (used as ID prefix).
        chunks: List of chunk dicts from :mod:`pdf_processor`.
        file_hash: SHA-256 of the source file, for duplicate detection.
        num_pages: Number of pages in the source document.
        uploaded_at: ISO timestamp of the upload.
        file_path: Path of the stored upload.

    Returns:
        Number of chunks added.
    """
    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{doc_id}_chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append(
            {
                "document_name": chunk["document_name"],
                "page_number": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "doc_id": doc_id,
                "file_hash": file_hash,
                "num_pages": num_pages,
                "uploaded_at": uploaded_at,
                "file_path": file_path,
            }
        )

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Added %d chunks for document '%s'", len(ids), doc_id)

    return len(ids)


def query(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve the most relevant *unique* chunks for a question.

    A candidate pool larger than ``top_k`` is fetched first so that duplicate
    chunks (e.g. from the same book uploaded more than once) do not consume
    all of the available slots.  Results are then de-duplicated by text and
    weak, off-topic matches are filtered out.

    Args:
        question: The user's natural-language question.
        top_k: Number of results to return (default from settings).

    Returns:
        List of dicts with keys ``text``, ``document_name``,
        ``page_number``, ``chunk_index``, and ``distance``.
    """
    top_k = top_k or settings.TOP_K
    collection = _get_collection()

    total = collection.count()
    if total == 0:
        return []

    candidate_count = min(
        total,
        max(top_k, 1) * max(settings.RETRIEVAL_CANDIDATE_MULTIPLIER, 1),
    )
    results = collection.query(
        query_texts=[question],
        n_results=candidate_count,
    )

    candidates: list[dict[str, Any]] = []
    if results and results["documents"] and results["documents"][0]:
        for idx, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][idx] if results["metadatas"] else {}  # type: ignore[index]
            distance = (
                results["distances"][0][idx] if results["distances"] else 1.0  # type: ignore[index]
            )
            candidates.append(
                {
                    "text": doc_text,
                    "document_name": meta.get("document_name", "unknown"),
                    "page_number": meta.get("page_number", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "doc_id": meta.get("doc_id", ""),
                    "distance": distance,
                }
            )

    candidates.sort(key=lambda hit: hit["distance"])

    deduped: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    seen_locations: set[tuple[str, int, int]] = set()
    for hit in candidates:
        location = (
            hit["document_name"],
            hit["page_number"],
            hit["chunk_index"],
        )
        text_key = _text_key(hit["text"])
        if text_key in seen_texts or location in seen_locations:
            continue
        seen_texts.add(text_key)
        seen_locations.add(location)
        deduped.append(hit)

    # No similarity cutoff here: weak matches are still passed to the model,
    # which decides whether they are sufficient. rag_engine logs every hit.
    hits = deduped[:top_k]

    logger.info(
        "Retrieved %d unique chunks from %d candidates for: %s",
        len(hits),
        len(candidates),
        question[:120],
    )
    return hits


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a specific document.

    Args:
        doc_id: The document identifier whose chunks should be deleted.
    """
    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})
    logger.info("Deleted all chunks for document '%s'", doc_id)


def find_document_by_hash(file_hash: str) -> str | None:
    """Return the doc_id of an already-indexed file with the same hash."""
    if not file_hash:
        return None
    collection = _get_collection()
    if collection.count() == 0:
        return None
    data = collection.get(
        where={"file_hash": file_hash},
        include=["metadatas"],
        limit=1,
    )
    if data["metadatas"]:
        return str(data["metadatas"][0].get("doc_id", "")) or None
    return None


def get_document_metadata() -> list[dict[str, Any]]:
    """Rebuild document metadata from the vector store.

    This keeps the document list accurate after an application restart, when
    the in-memory registry in :mod:`app.routes.documents` is empty.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    data = collection.get(include=["metadatas"])
    grouped: dict[str, dict[str, Any]] = {}
    for meta in data["metadatas"] or []:
        if not meta:
            continue
        doc_id = str(meta.get("doc_id", ""))
        if not doc_id:
            continue
        entry = grouped.setdefault(
            doc_id,
            {
                "id": doc_id,
                "name": meta.get("document_name", "unknown"),
                "pages": int(meta.get("num_pages", 0) or 0),
                "chunks": 0,
                "uploaded_at": meta.get("uploaded_at", ""),
                "file_path": meta.get("file_path", ""),
                "file_hash": meta.get("file_hash", ""),
            },
        )
        entry["chunks"] += 1
        entry["pages"] = max(entry["pages"], int(meta.get("page_number", 0) or 0))

    return sorted(grouped.values(), key=lambda item: item["uploaded_at"])


def deduplicate_store() -> int:
    """Delete duplicate chunk embeddings, keeping one copy of each text.

    Returns the number of embeddings removed.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return 0

    data = collection.get(include=["documents"])
    ids: list[str] = data["ids"] or []
    documents: list[str] = data["documents"] or []

    keep: set[str] = set()
    remove: list[str] = []
    for chunk_id, text in zip(ids, documents):
        key = _text_key(text or "")
        if key in keep:
            remove.append(chunk_id)
        else:
            keep.add(key)

    if remove:
        for start in range(0, len(remove), 500):
            collection.delete(ids=remove[start : start + 500])
        logger.info("Removed %d duplicate embeddings", len(remove))

    return len(remove)


def clear_collection() -> None:
    """Remove every embedding from the collection."""
    collection = _get_collection()
    ids = collection.get(include=[])["ids"] or []
    if ids:
        for start in range(0, len(ids), 500):
            collection.delete(ids=ids[start : start + 500])
    logger.info("Cleared collection '%s' (%d embeddings)", COLLECTION_NAME, len(ids))
