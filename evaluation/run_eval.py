"""
Evaluation runner script.

Run from the project root:
    python -m evaluation.run_eval

Loads questions from eval_questions.json, runs them through the RAG
engine, and prints per-question scores plus overall accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``app`` is importable.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.services.evaluator import load_eval_questions, run_evaluation  # noqa: E402
from app.services.vector_store import init_vector_store  # noqa: E402


def main() -> None:
    """Run the evaluation pipeline and print a report."""
    print("=" * 60)
    print("  📄 DocChat — Evaluation Runner")
    print("=" * 60)

    # Initialise the vector store (required before querying)
    init_vector_store()

    # Load questions
    questions = load_eval_questions()
    print(f"\nLoaded {len(questions)} evaluation questions.\n")

    # Run evaluation
    report = run_evaluation(questions)

    # Print results
    for i, result in enumerate(report["results"], start=1):
        status = "✅" if result.score >= 0.5 else "❌"
        print(f"{status} Q{i}: {result.question}")
        print(f"     Expected : {result.expected[:80]}…")
        print(f"     Actual   : {result.actual[:80]}…")
        print(f"     Score    : {result.score}  |  Sources found: {result.sources_found}")
        print()

    print("-" * 60)
    print(f"Overall Accuracy: {report['accuracy']:.1%}  ({report['total_questions']} questions)")
    print("=" * 60)


if __name__ == "__main__":
    main()
