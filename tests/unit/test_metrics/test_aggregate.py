"""Tests for cross-result aggregation (prefill curve, cold/warm ratio)."""

from __future__ import annotations

from typing import Any

from airllm_bench.harness.result import RunResult
from airllm_bench.metrics.aggregate import cold_warm_ratio, ttft_vs_length

_BASE: dict[str, Any] = {
    "exp_id": "e",
    "runner": "airllm",
    "quant": "nf4",
    "model": "m",
    "param_count": 1,
    "prompt_tokens": 32,
    "max_new_tokens": 4,
    "phase": "cold",
    "ok": True,
}


def _result(**kw: Any) -> RunResult:
    return RunResult(**{**_BASE, **kw})


def test_ttft_vs_length_excludes_missing() -> None:
    rs = [
        _result(prompt_tokens=32, ttft_s=1.0),
        _result(prompt_tokens=256, ttft_s=3.0),
        _result(prompt_tokens=1024, ok=False, ttft_s=None),
    ]
    assert ttft_vs_length(rs) == {32: 1.0, 256: 3.0}


def test_cold_warm_ratio() -> None:
    cold = _result(phase="cold", tpot_s=20.0)
    warm = _result(phase="warm", tpot_s=10.0)
    assert cold_warm_ratio(cold, warm) == 2.0


def test_cold_warm_ratio_none_when_missing() -> None:
    assert cold_warm_ratio(_result(tpot_s=None), _result(tpot_s=10.0)) is None
