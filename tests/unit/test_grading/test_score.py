"""Tests for the keyless self-grade scorer (T7.1).

``self_grade.py`` is the runnable, grader-facing gate behind ``SELF_GRADE.md``:
it reads the rubric from committed config (no hardcoded weights — D5/§4), does the
share-weighted arithmetic, and fails loud on an invalid or over-cap grade. The
rubric *scores* are a human judgment finalized in T7.3; the *math + invariants*
are tested here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from airllm_bench.grading.score import Rubric, load_rubric, validate_rubric, weighted_total

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _rubric(categories: list[dict[str, Any]], cap: float = 95.0) -> Rubric:
    payload = {"version": "1.00", "cap": cap, "categories": categories, "raw_data": {}}
    return Rubric.model_validate(payload)


def test_weighted_total_is_a_share_weighted_mean() -> None:
    rubric = _rubric(
        [
            {"name": "a", "weight": 50, "score_pct": 90},
            {"name": "b", "weight": 50, "score_pct": 100},
        ]
    )
    assert weighted_total(rubric) == pytest.approx(95.0)


def test_a_clean_rubric_has_no_problems() -> None:
    assert validate_rubric(_rubric([{"name": "a", "weight": 100, "score_pct": 92}])) == []


def test_weights_not_summing_to_100_are_flagged() -> None:
    problems = validate_rubric(_rubric([{"name": "a", "weight": 40, "score_pct": 90}]))
    assert any("weight" in problem.lower() for problem in problems)


def test_a_total_above_the_cap_is_flagged() -> None:
    problems = validate_rubric(_rubric([{"name": "a", "weight": 100, "score_pct": 99}], cap=95))
    assert any("cap" in problem.lower() for problem in problems)


def test_a_score_out_of_range_is_flagged() -> None:
    problems = validate_rubric(_rubric([{"name": "a", "weight": 100, "score_pct": 150}]))
    assert any("score" in problem.lower() for problem in problems)


def test_fractional_weights_summing_to_100_within_float_error_pass() -> None:
    # 1.1 accumulated 90× + 1.0 == 100.00000000000001 in IEEE 754 — must not spuriously fail.
    weights = [1.1] * 90 + [1.0]
    assert sum(weights) != 100.0  # guard: this case really does trip strict equality
    rubric = _rubric(
        [{"name": str(i), "weight": w, "score_pct": 90} for i, w in enumerate(weights)]
    )
    assert not any("weight" in problem.lower() for problem in validate_rubric(rubric))


def test_a_negative_weight_is_flagged() -> None:
    rubric = _rubric(
        [
            {"name": "a", "weight": -10, "score_pct": 90},
            {"name": "b", "weight": 110, "score_pct": 90},
        ]
    )
    problems = validate_rubric(rubric)
    assert any("weight" in problem.lower() and "a" in problem for problem in problems)


def test_the_committed_rubric_loads_validates_and_lands_in_range() -> None:
    rubric = load_rubric(_REPO_ROOT / "config" / "self_grade.json")
    assert validate_rubric(rubric) == []
    assert 92.0 <= weighted_total(rubric) <= 95.0
