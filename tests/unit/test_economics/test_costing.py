"""Tests for API token costing (the lifted cost_of, enforcement-free)."""

from __future__ import annotations

import pytest

from airllm_bench.economics.costing import TokenRate, cost_api, cost_api_cached


def test_cost_api_bills_input_and_output_separately() -> None:
    rate = TokenRate(input_per_mtok=0.2, output_per_mtok=0.4)
    assert cost_api(1_000_000, 0, rate) == pytest.approx(0.2)
    assert cost_api(0, 1_000_000, rate) == pytest.approx(0.4)
    assert cost_api(500_000, 500_000, rate) == pytest.approx(0.3)


def test_cost_api_zero_tokens() -> None:
    rate = TokenRate(input_per_mtok=1.0, output_per_mtok=1.0)
    assert cost_api(0, 0, rate) == 0.0


def test_cached_input_is_discounted() -> None:
    rate = TokenRate(input_per_mtok=1.0, output_per_mtok=2.0)
    # 1M cached x 0.1 = 0.1 ; 1M fresh x 1.0 = 1.0 ; 1M out x 2.0 = 2.0  => 3.1
    assert cost_api_cached(1_000_000, 1_000_000, 1_000_000, rate, 0.1) == pytest.approx(3.1)


def test_cached_with_no_cache_equals_plain() -> None:
    rate = TokenRate(input_per_mtok=1.5, output_per_mtok=0.5)
    plain = cost_api(1_000_000, 200_000, rate)
    cached = cost_api_cached(0, 1_000_000, 200_000, rate, 0.1)
    assert cached == pytest.approx(plain)
