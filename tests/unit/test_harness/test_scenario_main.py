"""Keyless test for the subprocess entrypoint (fake GPU stack + mock runner)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from airllm_bench.harness import scenario_main

from ._hwfakes import fake_psutil, fake_pynvml


def _write_configs(write_config: Callable[..., Path]) -> tuple[Path, Path]:
    setup = write_config(
        {
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
            "paths": {"layer_shards_saving_path": "C:/s"},
        },
        name="setup.json",
    )
    exp = write_config(
        {
            "version": "1.00",
            "runner": "mock",
            "quant": "none",
            "prompt": "what is virtual memory",
            "max_new_tokens": 3,
            "phase": "cold",
        },
        name="exp.json",
    )
    return exp, setup


def test_main_runs_and_persists(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml())
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil())
    exp, setup = _write_configs(write_config)
    results = tmp_path / "results"
    code = scenario_main.main([str(exp), str(setup), "exp1", str(results)])
    assert code == 0
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in (results / "exp1.jsonl").read_text("utf-8").splitlines()
    ]
    assert rows[0]["exp_id"] == "exp1"
    assert rows[0]["ok"] is True
    assert rows[0]["peak_vram_mb"] == pytest.approx(6_700.0)  # sampled via the fake reader
