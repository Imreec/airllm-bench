"""Tests for per-token energy derivation (measured GPU + estimated CPU)."""

from __future__ import annotations

import pytest

from airllm_bench.economics.energy import (
    avg_power_w,
    cpu_energy_per_token_j,
    energy_per_token_j,
    gpu_energy_per_token_j,
)
from airllm_bench.harness.result import RunResult


def _result(**over: object) -> RunResult:
    base: dict[str, object] = {
        "exp_id": "e",
        "runner": "airllm",
        "quant": "nf4",
        "model": "m",
        "param_count": 1,
        "prompt_tokens": 40,
        "max_new_tokens": 10,
        "phase": "warm",
        "ok": True,
        "tpot_s": 2.0,
        "gpu_energy_j": 1000.0,
        "runtime_s": 100.0,
    }
    base.update(over)
    return RunResult.model_validate(base)


def test_avg_power_is_energy_over_runtime() -> None:
    assert avg_power_w(1000.0, 100.0) == pytest.approx(10.0)


def test_gpu_energy_per_token_is_avg_power_times_tpot() -> None:
    # avg power 10 W over the run x 2 s/token = 20 J/token of decode energy.
    assert gpu_energy_per_token_j(_result()) == pytest.approx(20.0)


def test_cpu_energy_per_token_scales_tdp_by_load_fraction() -> None:
    # 100 W TDP x 0.5 load x 2 s/token = 100 J/token.
    assert cpu_energy_per_token_j(2.0, cpu_tdp_w=100.0, cpu_load_fraction=0.5) == pytest.approx(
        100.0
    )


def test_energy_per_token_sums_measured_gpu_and_estimated_cpu() -> None:
    total = energy_per_token_j(_result(), cpu_tdp_w=100.0, cpu_load_fraction=0.5)
    assert total == pytest.approx(120.0)


def test_energy_per_token_is_none_without_timing() -> None:
    assert energy_per_token_j(_result(ok=False, tpot_s=None), 100.0, 0.5) is None
