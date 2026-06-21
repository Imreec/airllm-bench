"""Tests for the five-line break-even model (fixed CAPEX + marginal cost)."""

from __future__ import annotations

from typing import Any

import pytest

from airllm_bench.economics.breakeven import (
    Line,
    breakeven_volume,
    build_lines,
    cumulative_cost,
    curve,
)
from airllm_bench.economics.config import EconomicsConfig
from airllm_bench.harness.result import RunResult


def _cfg(data: dict[str, Any]) -> EconomicsConfig:
    return EconomicsConfig.model_validate({**data, "raw_data": {}})


def _result(exp_id: str, **over: object) -> RunResult:
    base: dict[str, object] = {
        "exp_id": exp_id,
        "runner": "llamacpp",
        "quant": "q4",
        "model": "m",
        "param_count": 1,
        "prompt_tokens": 40,
        "max_new_tokens": 10,
        "phase": "warm",
        "ok": True,
        "tpot_s": 2.0,
        "throughput_tok_s": 0.5,
        "gpu_energy_j": 1000.0,
        "runtime_s": 100.0,
    }
    base.update(over)
    return RunResult.model_validate(base)


def test_cumulative_cost_is_fixed_plus_marginal() -> None:
    line = Line(name="x", fixed_usd=100.0, marginal_usd_per_token=1e-6)
    assert cumulative_cost(line, 0) == pytest.approx(100.0)
    assert cumulative_cost(line, 1_000_000) == pytest.approx(101.0)


def test_curve_is_monotonic_non_decreasing() -> None:
    line = Line(name="x", fixed_usd=100.0, marginal_usd_per_token=1e-6)
    ys = curve(line, [0, 1_000_000, 5_000_000])
    assert ys == sorted(ys)


def test_breakeven_volume_is_where_two_lines_cross() -> None:
    fixed = Line(name="onprem", fixed_usd=100.0, marginal_usd_per_token=0.0)
    api = Line(name="api", fixed_usd=0.0, marginal_usd_per_token=1e-6)
    # 100 fixed / 1e-6 per token = 1e8 tokens.
    assert breakeven_volume(fixed, api) == pytest.approx(1e8)


def test_breakeven_volume_none_when_slopes_equal() -> None:
    a = Line(name="a", fixed_usd=0.0, marginal_usd_per_token=1e-6)
    b = Line(name="b", fixed_usd=50.0, marginal_usd_per_token=1e-6)
    assert breakeven_volume(a, b) is None


def test_breakeven_volume_none_when_intersection_not_positive() -> None:
    # b dominates a (higher fixed AND higher marginal) -> they only "cross" at v<0,
    # i.e. a never becomes cheaper in feasible volume. No break-even.
    a = Line(name="a", fixed_usd=0.0, marginal_usd_per_token=1e-6)
    b = Line(name="b", fixed_usd=100.0, marginal_usd_per_token=2e-6)
    assert breakeven_volume(b, a) is None


def test_build_lines_produces_the_five_canonical_lines(
    economics_config_data: dict[str, Any],
) -> None:
    lines = build_lines(
        _cfg(economics_config_data),
        model_key="qwen2.5-32b",
        realistic=_result("llamacpp-q4-warm"),
        airllm=_result("airllm-nf4-warm", runner="airllm", tpot_s=20.0),
        input_per_output_ratio=1.0,
    )
    assert set(lines) == {"api", "api_cached", "onprem_realistic", "onprem_airllm", "cloud"}
    # On-prem carries the up-front CAPEX as fixed; API/cloud are pay-as-you-go.
    assert lines["onprem_realistic"].fixed_usd == pytest.approx(1200.0)
    assert lines["api"].fixed_usd == 0.0
    assert lines["cloud"].fixed_usd == 0.0
    # Caching lowers the API marginal (discounted input share).
    assert lines["api_cached"].marginal_usd_per_token < lines["api"].marginal_usd_per_token
    # AirLLM's slow decode burns more energy per token than the realistic competitor.
    assert (
        lines["onprem_airllm"].marginal_usd_per_token
        > lines["onprem_realistic"].marginal_usd_per_token
    )


def test_stalled_run_makes_cloud_infinitely_expensive_not_free(
    economics_config_data: dict[str, Any],
) -> None:
    # A zero/None throughput must read as infinite $/token, never as $0 (free).
    lines = build_lines(
        _cfg(economics_config_data),
        model_key="qwen2.5-32b",
        realistic=_result("stalled", throughput_tok_s=None),
        airllm=_result("airllm-nf4-warm", runner="airllm", tpot_s=20.0),
        input_per_output_ratio=1.0,
    )
    assert lines["cloud"].marginal_usd_per_token == float("inf")
