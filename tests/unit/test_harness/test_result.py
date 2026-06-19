"""Tests for the RunResult schema (the analysis source of truth)."""

from __future__ import annotations

import json

from airllm_bench.harness.result import SCHEMA_VERSION, RunResult

_BASE = {
    "exp_id": "e",
    "runner": "mock",
    "quant": "none",
    "model": "m",
    "param_count": 1,
    "prompt_tokens": 3,
    "max_new_tokens": 4,
    "phase": "cold",
}


def test_minimal_ok_result_defaults() -> None:
    r = RunResult(ok=True, **_BASE)
    assert r.schema_version == SCHEMA_VERSION
    assert r.itl_s == []
    assert r.error is None
    assert r.ttft_s is None


def test_baseline_oom_case() -> None:
    r = RunResult(ok=False, error="OutOfMemoryError", **{**_BASE, "runner": "baseline_hf"})
    assert not r.ok
    assert r.error == "OutOfMemoryError"
    assert r.perplexity is None


def test_json_roundtrip_preserves_fields() -> None:
    r = RunResult(
        ok=True,
        ttft_s=1.2,
        itl_s=[0.1, 0.2],
        tpot_s=0.15,
        throughput_tok_s=6.0,
        peak_vram_mb=6700.0,
        gpu_energy_j=150.0,
        perplexity=7.8,
        **{**_BASE, "quant": "nf4"},
    )
    blob = r.model_dump_json()
    assert RunResult.model_validate_json(blob) == r
    assert json.loads(blob)["itl_s"] == [0.1, 0.2]
