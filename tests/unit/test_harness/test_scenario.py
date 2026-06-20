"""Keyless tests for the single-scenario worker (mock runner, no GPU)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from airllm_bench.harness.scenario import run_scenario
from airllm_bench.shared.config_models import ExperimentConfig, SetupConfig


def _setup(write_config: Callable[..., Path]) -> SetupConfig:
    data = {
        "version": "1.00",
        "hardware": {
            "cpu": "x",
            "cores": 1,
            "gpu": "g",
            "vram_gb": 12,
            "ram_gb": 32,
            "nvme_path": "C:/s",
        },
        "model": {"repo_id": "Qwen/Qwen2.5-32B-Instruct", "params": 32_000_000_000},
        "paths": {"layer_shards_saving_path": "C:/airllm_shards"},
    }
    return SetupConfig.from_file(write_config(data, name="setup.json"))


def _exp(write_config: Callable[..., Path], **over: Any) -> ExperimentConfig:
    data = {
        "version": "1.00",
        "runner": "mock",
        "quant": "none",
        "prompt": "what is virtual memory paging",
        "max_new_tokens": 3,
        "phase": "cold",
        **over,
    }
    return ExperimentConfig.from_file(write_config(data, name="exp.json"))


def test_run_scenario_returns_ok_result(write_config: Callable[..., Path]) -> None:
    result = run_scenario(_exp(write_config), _setup(write_config), exp_id="s1")
    assert result.ok
    assert result.exp_id == "s1"
    assert result.prompt_tokens == 5  # whitespace estimate of the 5-word prompt


def test_pinned_prompt_tokens_override(write_config: Callable[..., Path]) -> None:
    result = run_scenario(_exp(write_config, prompt_tokens=256), _setup(write_config), exp_id="s2")
    assert result.prompt_tokens == 256  # the matrix-pinned native count wins


def test_run_scenario_writes_jsonl(write_config: Callable[..., Path], tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    result = run_scenario(
        _exp(write_config), _setup(write_config), exp_id="s3", results_dir=results_dir
    )
    path = results_dir / "s3.jsonl"
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["exp_id"] == "s3"
    assert rows[0]["schema_version"] == result.schema_version


def test_append_accumulates_lines(write_config: Callable[..., Path], tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for _ in range(2):
        run_scenario(_exp(write_config), _setup(write_config), exp_id="s4", results_dir=results_dir)
    lines = (results_dir / "s4.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # appended, not overwritten
