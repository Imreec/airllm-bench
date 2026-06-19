"""Tests for perplexity."""

from __future__ import annotations

import pytest

from airllm_bench.metrics.quality import perplexity


def test_uniform_logits_perplexity_equals_vocab_size() -> None:
    # Uniform over a vocab of 2 -> ppl = 2 regardless of which token is the target.
    logits = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert perplexity(logits, [0, 1, 0]) == pytest.approx(2.0)


def test_confident_correct_prediction_near_one() -> None:
    # logits[0] strongly favors token 0, and the target (token_ids[1]) is 0.
    logits = [[10.0, 0.0], [0.0, 10.0]]
    assert perplexity(logits, [1, 0]) == pytest.approx(1.0, abs=1e-3)


def test_too_few_tokens_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        perplexity([[0.0, 0.0]], [0])
