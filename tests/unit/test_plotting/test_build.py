"""Integration-ish test: the figure pipeline runs on committed data, keyless."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from airllm_bench.harness.result import RunResult
from airllm_bench.plotting.build import build_figures

REPO_ROOT = Path(__file__).resolve().parents[3]
_SETUP = REPO_ROOT / "config" / "setup.json"
_ECON = REPO_ROOT / "config" / "economics.json"
_EXPECTED = {
    "ttft_vs_length.png",
    "itl.png",
    "cold_warm_ttft.png",
    "throughput.png",
    "perplexity.png",
    "roofline.png",
    "breakeven.png",
}


def test_build_figures_writes_every_expected_png(tmp_path: Path) -> None:
    paths = build_figures(REPO_ROOT / "results", _SETUP, _ECON, tmp_path)
    assert {p.name for p in paths} == _EXPECTED
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_build_figures_raises_without_a_roofline_block(
    tmp_path: Path, setup_config_data: dict[str, Any]
) -> None:
    bare = tmp_path / "setup.json"  # the conftest setup has no roofline block
    bare.write_text(json.dumps(setup_config_data), encoding="utf-8")
    with pytest.raises(ValueError, match="roofline"):
        build_figures(REPO_ROOT / "results", bare, _ECON, tmp_path / "out")


def test_build_figures_skips_breakeven_without_economics_runs(tmp_path: Path) -> None:
    # A results dir with no llama.cpp/AirLLM runs -> no on-prem lines -> no break-even.
    res = tmp_path / "results"
    res.mkdir()
    row = RunResult(
        exp_id="mock-only",
        runner="mock",
        quant="none",
        model="m",
        param_count=1,
        prompt_tokens=40,
        max_new_tokens=10,
        phase="warm",
        ok=True,
    )
    (res / "mock-only.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8")
    paths = build_figures(res, _SETUP, _ECON, tmp_path / "out")
    assert "breakeven.png" not in {p.name for p in paths}
    assert len(paths) == 6
