"""Keyless tests for the matrix orchestrator (spawn + flush injected)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from airllm_bench.harness.orchestrator import run_matrix


def _setup_path(write_config: Callable[..., Path]) -> Path:
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
        "model": {"repo_id": "m", "params": 1},
        "paths": {"layer_shards_saving_path": "C:/s"},
    }
    return write_config(data, name="setup.json")


def _scenario(write_config: Callable[..., Path], phase: str, name: str) -> Path:
    data = {
        "version": "1.00",
        "runner": "mock",
        "quant": "none",
        "prompt": "hi",
        "phase": phase,
    }
    return write_config(data, name=name)


def test_runs_each_scenario_in_order(write_config: Callable[..., Path], tmp_path: Path) -> None:
    spawned: list[str] = []

    def spawn(exp: Path, setup: Path, exp_id: str, results: Path) -> int:
        spawned.append(exp_id)
        return 0

    paths = [
        _scenario(write_config, "cold", "a.json"),
        _scenario(write_config, "warm", "b.json"),
    ]
    codes = run_matrix(
        paths,
        _setup_path(write_config),
        results_dir=tmp_path,
        spawn=spawn,
        progress=lambda x: x,
    )
    assert codes == [0, 0]
    assert spawned == ["a", "b"]  # order preserved (cold then its warm twin)


def test_flushes_before_cold_only(write_config: Callable[..., Path], tmp_path: Path) -> None:
    flushes = {"n": 0}

    def flush() -> None:
        flushes["n"] += 1

    paths = [
        _scenario(write_config, "cold", "c.json"),
        _scenario(write_config, "warm", "w.json"),
    ]
    run_matrix(
        paths,
        _setup_path(write_config),
        results_dir=tmp_path,
        spawn=lambda *a: 0,
        cold_flush=flush,
        progress=lambda x: x,
    )
    assert flushes["n"] == 1  # flushed before the cold scenario, not the warm one


def test_propagates_nonzero_exit_codes(write_config: Callable[..., Path], tmp_path: Path) -> None:
    def spawn(exp: Path, setup: Path, exp_id: str, results: Path) -> int:
        return 3

    codes = run_matrix(
        [_scenario(write_config, "warm", "x.json")],
        _setup_path(write_config),
        results_dir=tmp_path,
        spawn=spawn,
        progress=lambda x: x,
    )
    assert codes == [3]
