"""Tests for timing derivation and ITL statistics."""

from __future__ import annotations

import pytest

from airllm_bench.metrics.timing import compute_timing, itl_stats


def test_empty_token_times() -> None:
    t = compute_timing(0.0, [])
    assert t.ttft_s is None
    assert t.itl_s == []
    assert t.tpot_s is None
    assert t.throughput_tok_s is None


def test_basic_split() -> None:
    t = compute_timing(1.0, [2.0, 3.0, 4.5])  # gen start 1.0
    assert t.ttft_s == pytest.approx(1.0)
    assert t.itl_s == pytest.approx([1.0, 1.5])
    assert t.tpot_s == pytest.approx(1.25)
    assert t.throughput_tok_s == pytest.approx(3 / 3.5)


def test_single_token_has_no_itl() -> None:
    t = compute_timing(0.0, [2.0])
    assert t.ttft_s == pytest.approx(2.0)
    assert t.itl_s == []
    assert t.tpot_s is None
    assert t.throughput_tok_s == pytest.approx(0.5)


def test_itl_stats() -> None:
    s = itl_stats([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == pytest.approx(2.5)
    assert s["min"] == 1.0
    assert s["max"] == 4.0


def test_itl_stats_empty() -> None:
    assert itl_stats([]) == {}
