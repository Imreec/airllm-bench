"""Keyless tests for the thin bench CLI (orchestrator + subprocess mocked)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from airllm_bench import cli


def _setup_path(write_config: Callable[..., Path]) -> Path:
    return write_config(
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
            "model": {"repo_id": "m", "params": 1},
            "paths": {"layer_shards_saving_path": "C:/s"},
            "cold_flush": {"command": ["RAMMap64.exe", "-Et"]},
        },
        name="setup.json",
    )


def _args(setup: Path, *, no_flush: bool) -> Any:
    argv = ["bench", "--experiments", "a.json", "b.json", "--setup", str(setup)]
    if no_flush:
        argv.append("--no-flush")
    return cli.build_parser().parse_args(argv)


def test_no_flush_passes_none(write_config: Callable[..., Path]) -> None:
    seen: dict[str, Any] = {}

    def orchestrate(paths: list[Path], setup: Path, **kw: Any) -> list[int]:
        seen.update(kw, paths=paths)
        return [0, 0]

    code = cli._bench(_args(_setup_path(write_config), no_flush=True), orchestrate=orchestrate)
    assert code == 0
    assert seen["cold_flush"] is None
    assert len(seen["paths"]) == 2


def test_flush_built_when_enabled(write_config: Callable[..., Path]) -> None:
    seen: dict[str, Any] = {}

    def orchestrate(paths: list[Path], setup: Path, **kw: Any) -> list[int]:
        seen.update(kw)
        return [0]

    cli._bench(_args(_setup_path(write_config), no_flush=False), orchestrate=orchestrate)
    assert callable(seen["cold_flush"])  # a real cold-flush closure was wired in


def test_nonzero_code_when_a_scenario_fails(write_config: Callable[..., Path]) -> None:
    code = cli._bench(
        _args(_setup_path(write_config), no_flush=True),
        orchestrate=lambda paths, setup, **kw: [0, 1],
    )
    assert code == 1


def test_spawn_invokes_scenario_module(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], check: bool) -> SimpleNamespace:
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    code = cli._spawn_subprocess(Path("e.json"), Path("s.json"), "id1", Path("res"))
    assert code == 0
    assert cli._SCENARIO_MODULE in captured["cmd"]
    assert "id1" in captured["cmd"]


def test_main_dispatches_to_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_bench", lambda args: 7)
    assert cli.main(["bench", "--experiments", "x.json"]) == 7
