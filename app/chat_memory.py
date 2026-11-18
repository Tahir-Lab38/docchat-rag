"""
Terminal PDF chatbot with RAG-style retrieval.

Asks for a PDF path (or pasted text), chunks it, and answers questions
using retrieved context via Groq. Run: python -m app.chat_memory
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from groq import Groq

from app.config import settings
from app.prompts.templates import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from app.services import pdf_processor


def create_client() -> Groq:
    if not settings.GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)
    return Groq(api_key=settings.GROQ_API_KEY)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_chunks(question: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not chunks:
        return []
    q_tokens = _tokenize(question)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, chunk in enumerate(chunks):
        c_tokens = _tokenize(chunk["text"])
        score = len(q_tokens & c_tokens)
        scored.append((score, idx, chunk))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    top = [c for s, _, c in scored[:top_k] if s > 0]
    if not top:
        top = chunks[: min(top_k, len(chunks))]
    return top


def build_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(No relevant context found.)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[Source {i}: {chunk.get('document_name','document')}, Page {chunk.get('page', 0)}]"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n".join(parts)


def answer_question(
    client: Groq,
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> str:
    relevant = retrieve_chunks(question, chunks, settings.TOP_K)
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=build_context(relevant),
        question=question,
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def load_pdf(path: str) -> list[dict[str, Any]]:
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    pages = pdf_processor.extract_text_from_pdf(path)
    name = os.path.basename(path)
    chunks = pdf_processor.chunk_text(pages, document_name=name)
    if not chunks:
        raise ValueError("No readable text found in the PDF.")
    return chunks


def load_pasted_text() -> list[dict[str, Any]]:
    print("Paste your text/book content. End with a line containing only: END_PASTE")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END_PASTE":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("No text provided.")
    pages = [{"page": 1, "text": text}]
    chunks = pdf_processor.chunk_text(pages, document_name="pasted_text")
    if not chunks:
        raise ValueError("No readable text found.")
    return chunks


def main() -> None:
    client = create_client()
    print("╔══════════════════════════════════════════╗")
    print("║     DocChat — Terminal PDF Chatbot       ║")
    print("║     Type 'quit' or 'exit' to leave       ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("Provide a PDF to chat with:")
    print("  - Enter a PDF file path")
    print("  - Type 'paste' to paste text (end with END_PASTE)")
    print("  - Press Enter to skip (no document context)")
    print()

    chunks: list[dict[str, Any]] = []
    try:
        source = input("PDF path / paste / Enter: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        return

    try:
        if source.lower() == "paste":
            chunks = load_pasted_text()
            print(f"\nLoaded pasted text ({len(chunks)} chunks).\n")
        elif source:
            chunks = load_pdf(source)
            name = os.path.basename(source.strip().strip('"').strip("'"))
            print(f"\nLoaded '{name}' ({len(chunks)} chunks).\n")
        else:
            print("\nNo document loaded. Answers will be general (no citations).\n")
    except Exception as exc:
        print(f"\n[Error loading document] {exc}\nStarting without document context.\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            if chunks:
                reply = answer_question(client, user_input, chunks, history)
            else:
                history.append({"role": "user", "content": user_input})
                resp = client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[{"role": "system", "content": "You are DocChat, a helpful AI assistant."}] + history,
                    temperature=0.7,
                    max_tokens=1024,
                )
                reply = (resp.choices[0].message.content or "").strip()
                history.append({"role": "assistant", "content": reply})
            if chunks:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": reply})
            print(f"\nDocChat: {reply}\n")
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


if __name__ == "__main__":
    main()
