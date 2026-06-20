"""Keyless tests for the cold-cache flush + fail-loud admin guard."""

from __future__ import annotations

import pytest

from airllm_bench.harness.cache_flush import (
    NotElevatedError,
    build_cold_flush,
    flush_standby_list,
    require_admin,
)


def test_require_admin_passes_when_elevated() -> None:
    require_admin(lambda: True)  # no raise


def test_require_admin_fails_loud_when_not_elevated() -> None:
    with pytest.raises(NotElevatedError, match="Administrator"):
        require_admin(lambda: False)


def test_flush_invokes_the_tool() -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], check: bool) -> None:
        captured["cmd"] = cmd
        captured["check"] = check

    flush_standby_list(["RAMMap64.exe", "-accepteula", "-Et"], run=fake_run)
    assert captured["cmd"] == ["RAMMap64.exe", "-accepteula", "-Et"]
    assert captured["check"] is True


def test_build_cold_flush_runs_admin_then_flush() -> None:
    calls: list[list[str]] = []
    flush = build_cold_flush(
        {"command": ["RAMMap64.exe", "-Et"]},
        is_admin=lambda: True,
        run=lambda cmd, check: calls.append(cmd),
    )
    flush()
    assert calls == [["RAMMap64.exe", "-Et"]]  # admin passed, tool invoked once


def test_build_cold_flush_requires_admin() -> None:
    flush = build_cold_flush(
        {"command": ["RAMMap64.exe", "-Et"]},
        is_admin=lambda: False,
        run=lambda cmd, check: None,
    )
    with pytest.raises(NotElevatedError):
        flush()  # never reaches the flush tool
