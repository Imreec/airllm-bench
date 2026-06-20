"""Keyless tests for the cold-cache flush + fail-loud admin guard."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from airllm_bench.harness.cache_flush import (
    NotElevatedError,
    build_cold_flush,
    flush_standby_list,
    require_admin,
    verify_cache_dropped,
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

    flush_standby_list(["EmptyStandbyList.exe", "standbylist"], run=fake_run)
    assert captured["cmd"] == ["EmptyStandbyList.exe", "standbylist"]
    assert captured["check"] is True


@pytest.mark.parametrize(
    ("before", "after", "ok"), [(1000.0, 4000.0, True), (1000.0, 1500.0, False)]
)
def test_verify_cache_dropped(before: float, after: float, ok: bool) -> None:
    assert verify_cache_dropped(before, after, min_rise_mb=2000) is ok


def _avail_reader(values: list[float]) -> Callable[[], float]:
    it = iter(values)
    return lambda: next(it)


def test_build_cold_flush_happy_path() -> None:
    runs: list[list[str]] = []
    flush = build_cold_flush(
        {"command": ["tool"], "min_rise_mb": 2000},
        is_admin=lambda: True,
        run=lambda cmd, check: runs.append(cmd),
        read_avail_mb=_avail_reader([1000.0, 4000.0]),  # rose 3000 >= 2000
    )
    flush()
    assert runs == [["tool"]]


def test_build_cold_flush_raises_on_insufficient_drop() -> None:
    flush = build_cold_flush(
        {"command": ["tool"], "min_rise_mb": 2000},
        is_admin=lambda: True,
        run=lambda cmd, check: None,
        read_avail_mb=_avail_reader([1000.0, 1200.0]),  # only 200 freed
    )
    with pytest.raises(RuntimeError, match="freed only"):
        flush()


def test_build_cold_flush_requires_admin() -> None:
    flush = build_cold_flush(
        {"command": ["tool"]},
        is_admin=lambda: False,
        run=lambda cmd, check: None,
        read_avail_mb=_avail_reader([1000.0]),
    )
    with pytest.raises(NotElevatedError):
        flush()
