"""Keyless tests for the NVML+psutil telemetry reader (fakes injected in sys.modules)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from airllm_bench.harness.hw_reader import make_reader
from airllm_bench.harness.sampler import ResourceReading

_MB = 1024 * 1024


def _fake_pynvml() -> ModuleType:
    mod = ModuleType("pynvml")
    mod.nvmlInit = lambda: None  # type: ignore[attr-defined]
    mod.nvmlDeviceGetHandleByIndex = lambda i: f"handle{i}"  # type: ignore[attr-defined]
    mod.nvmlDeviceGetPowerUsage = lambda h: 157_000  # type: ignore[attr-defined] # mW
    mod.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(used=6_700 * _MB)  # type: ignore[attr-defined]
    return mod


def _fake_psutil() -> ModuleType:
    mod = ModuleType("psutil")
    mod.Process = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        memory_info=lambda: SimpleNamespace(rss=21_300 * _MB)
    )
    mod.virtual_memory = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        used=30_100 * _MB, available=1_900 * _MB
    )
    return mod


def test_reader_converts_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil())
    read = make_reader()
    reading = read()
    assert isinstance(reading, ResourceReading)
    assert reading.power_w == pytest.approx(157.0)  # mW -> W
    assert reading.vram_mb == pytest.approx(6_700.0)  # bytes -> MB
    assert reading.rss_mb == pytest.approx(21_300.0)
    assert reading.sys_used_mb == pytest.approx(30_100.0)
    assert reading.sys_avail_mb == pytest.approx(1_900.0)


def test_reader_honors_gpu_index(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}
    fake = _fake_pynvml()
    fake.nvmlDeviceGetHandleByIndex = lambda i: seen.setdefault("idx", i)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil())
    make_reader(gpu_index=0)
    assert seen["idx"] == 0
