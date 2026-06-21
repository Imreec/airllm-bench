"""Runnable self-grade gate behind SELF_GRADE.md (T7.1, D13).

Reads the rubric from ``config/self_grade.json``, prints the share-weighted
breakdown, and fails loud if the grade is internally inconsistent (weights ≠ 100,
a score out of 0–100, or a total above the self-imposed cap). The number is a human
judgment; this makes it re-computable and defensible by a grader. Wired into
``make grade`` once the report lands (T7.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

from airllm_bench.grading.score import load_rubric, validate_rubric, weighted_total

ROOT = Path(__file__).resolve().parent.parent
RUBRIC = ROOT / "config" / "self_grade.json"


def main() -> int:
    """Compute + validate the self-grade; print the breakdown and the total."""
    rubric = load_rubric(RUBRIC)
    for cat in rubric.categories:
        print(f"  {cat.weight:>5.0f}%  {cat.score_pct:>5.1f}  {cat.name}")
    total = weighted_total(rubric)
    print(f"\nSelf-grade: {total:.1f} / 100  (cap {rubric.cap:.0f})")
    problems = validate_rubric(rubric)
    for problem in problems:
        print(f"FAIL: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
