"""
Rebuild the vector index from a PDF using the current chunking settings.

Usage:
    python scripts/reindex.py "path/to/book.pdf"

The existing collection is cleared and replaced with freshly chunked embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services import pdf_processor, vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the PDF to index")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing index instead of replacing it",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        parser.error(f"PDF not found: {pdf_path}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    vector_store.init_vector_store()

    if not args.append:
        vector_store.clear_collection()
        print("Cleared existing vector index.")

    pages = pdf_processor.extract_text_from_pdf(str(pdf_path))
    chunks = pdf_processor.chunk_text(
        pages,
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP,
        document_name=pdf_path.name,
    )
    if not chunks:
        parser.error("No readable text found in the PDF.")

    doc_id = str(uuid.uuid4())
    stored_name = f"{doc_id}_{pdf_path.name}"
    stored_path = os.path.join(settings.UPLOAD_DIR, stored_name)
    shutil.copy2(pdf_path, stored_path)

    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    uploaded_at = datetime.now(timezone.utc).isoformat()

    added = vector_store.add_document(
        doc_id,
        chunks,
        file_hash=file_hash,
        num_pages=len(pages),
        uploaded_at=uploaded_at,
        file_path=stored_path,
    )

    print(f"Indexed '{pdf_path.name}'")
    print(f"  pages:  {len(pages)}")
    print(f"  chunks: {added}")
    print(f"  size:   {settings.CHUNK_SIZE} chars, {settings.CHUNK_OVERLAP} overlap")
    print(f"  total embeddings now: {vector_store.count()}")


if __name__ == "__main__":
    main()
