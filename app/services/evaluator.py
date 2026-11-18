"""
Evaluation service — run an evaluation set and compute scores.

Uses keyword matching plus an LLM-as-judge approach (Groq) to score
RAG answers against expected answers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.models.schemas import EvalResult
from app.prompts.templates import EVAL_JUDGE_PROMPT
from app.services import rag_engine
from app.services.rag_engine import get_groq_client

logger = logging.getLogger(__name__)

EVAL_FILE = Path(__file__).resolve().parent.parent.parent / "evaluation" / "eval_questions.json"


def _keyword_score(expected: str, actual: str) -> float:
    """Simple keyword overlap score between expected and actual answers.

    Returns a float in [0.0, 1.0].
    """
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())

    if not expected_words:
        return 1.0 if not actual_words else 0.0

    overlap = expected_words & actual_words
    return len(overlap) / len(expected_words)


def _llm_judge_score(question: str, expected: str, actual: str) -> float:
    """Ask Groq to score the actual answer against the expected one.

    Returns a float in [0.0, 1.0].  Falls back to 0.5 on error.
    """
    prompt = EVAL_JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        actual=actual,
    )

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        content: str = response.choices[0].message.content or ""  # type: ignore[union-attr]

        # Parse the JSON score from the response
        match = re.search(r'"score"\s*:\s*([\d.]+)', content)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # clamp
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)

    return 0.5  # fallback


def load_eval_questions(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the evaluation question set from a JSON file.

    Args:
        path: Path to the JSON file.  Defaults to ``evaluation/eval_questions.json``.

    Returns:
        List of question dicts.
    """
    eval_path = Path(path) if path else EVAL_FILE
    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_evaluation(
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the full evaluation pipeline.

    Args:
        questions: Optional list of question dicts. If not provided, loads
                   from the default evaluation file.

    Returns:
        Dict with ``results`` (list of :class:`EvalResult`) and ``accuracy`` (float).
    """
    if questions is None:
        questions = load_eval_questions()

    results: list[EvalResult] = []
    total_score = 0.0

    for item in questions:
        question = item["question"]
        expected = item["expected_answer"]
        expected_has_answer = item.get("expected_has_answer", True)

        # Run through the RAG engine
        rag_result = rag_engine.generate_answer(question)
        actual = rag_result["answer"]
        has_answer = rag_result["has_answer"]
        sources_found = len(rag_result["sources"]) > 0

        # Score: blend keyword + LLM-as-judge
        if not expected_has_answer:
            # For unanswerable questions, check if the model correctly refused
            score = 1.0 if not has_answer else 0.0
        else:
            kw_score = _keyword_score(expected, actual)
            llm_score = _llm_judge_score(question, expected, actual)
            score = 0.4 * kw_score + 0.6 * llm_score

        total_score += score
        results.append(
            EvalResult(
                question=question,
                expected=expected,
                actual=actual,
                score=round(score, 3),
                sources_found=sources_found,
            )
        )

    accuracy = round(total_score / len(results), 3) if results else 0.0

    return {
        "results": results,
        "accuracy": accuracy,
        "total_questions": len(results),
    }
