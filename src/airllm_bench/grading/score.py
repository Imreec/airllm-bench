"""Self-grade arithmetic + invariants (D13, §9).

The rubric (categories, weights, scores, cap) is *data* in ``config/self_grade.json``
— no grade numbers live in code (rule 5). This module does the share-weighted mean
and fails loud when the rubric is internally inconsistent: weights must total 100,
each score must be a 0–100 percentage, and the weighted total must not exceed the
self-imposed cap. The scores themselves are an honest human judgment (finalized in
T7.3); the scorer keeps them arithmetically defensible and re-runnable by a grader.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from airllm_bench.shared.config import load_versioned

#: Category weights must sum to exactly this (a full 100-point rubric).
_TOTAL_WEIGHT = 100.0


class Category(BaseModel):
    """One rubric row: its share of 100 (``weight``) and how well we scored it."""

    model_config = ConfigDict(extra="allow")

    name: str
    weight: float
    score_pct: float  # 0–100: performance within this category


class Rubric(BaseModel):
    """``config/self_grade.json`` — the weighted self-assessment rubric."""

    model_config = ConfigDict(extra="ignore")

    version: str
    cap: float
    categories: list[Category]
    raw_data: dict[str, object]

    @classmethod
    def from_file(cls, path: str | Path) -> Rubric:
        """Load and validate the committed rubric."""
        return load_versioned(path, cls)


def load_rubric(path: str | Path) -> Rubric:
    """Load ``config/self_grade.json`` (versioned, fail-loud)."""
    return Rubric.from_file(path)


def weighted_total(rubric: Rubric) -> float:
    """Share-weighted mean score = Σ (weight/100 · score_pct)."""
    return sum(cat.weight / _TOTAL_WEIGHT * cat.score_pct for cat in rubric.categories)


def validate_rubric(rubric: Rubric) -> list[str]:
    """Return every consistency problem (empty list ⇒ the grade is defensible)."""
    problems: list[str] = []
    weight_sum = sum(cat.weight for cat in rubric.categories)
    if weight_sum != _TOTAL_WEIGHT:
        problems.append(f"category weights sum to {weight_sum}, must be {_TOTAL_WEIGHT}")
    for cat in rubric.categories:
        if not 0.0 <= cat.score_pct <= 100.0:
            problems.append(f"score for '{cat.name}' is {cat.score_pct}, must be 0–100")
    total = weighted_total(rubric)
    if total > rubric.cap:
        problems.append(f"weighted total {total:.1f} exceeds the self-imposed cap {rubric.cap}")
    return problems
