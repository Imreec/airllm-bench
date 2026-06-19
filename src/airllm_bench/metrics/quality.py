"""Perplexity from teacher-forced logits (D10) — the primary quality metric.

Pure Python (works on list-of-lists from the mock *and* numpy/torch rows from a
real runner). ``logits[i]`` predicts ``token_ids[i+1]``; perplexity = exp(mean NLL).
Cheap on the AirLLM path: one forward pass, not autoregressive decode.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _neg_log_prob(row: Sequence[float], target: int) -> float:
    """-log softmax(row)[target], computed stably."""
    top = max(row)
    log_denom = math.log(sum(math.exp(x - top) for x in row))
    return -((row[target] - top) - log_denom)


def perplexity(logits: Sequence[Sequence[float]], token_ids: Sequence[int]) -> float:
    """exp(mean next-token NLL) over a teacher-forced sequence.

    Raises:
        ValueError: with fewer than 2 tokens (no next-token pairs to score).
    """
    n = min(len(logits), len(token_ids))
    if n < 2:
        msg = "perplexity needs at least 2 tokens"
        raise ValueError(msg)
    total_nll = sum(_neg_log_prob(logits[i], token_ids[i + 1]) for i in range(n - 1))
    return math.exp(total_nll / (n - 1))
