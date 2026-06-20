"""Keyless tests for the NVML+psutil telemetry reader (fakes injected in sys.modules)."""

from __future__ import annotations

import sys

import pytest

from airllm_bench.harness.hw_reader import make_reader
from airllm_bench.harness.sampler import ResourceReading

from ._hwfakes import fake_psutil, fake_pynvml


def test_reader_converts_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml())
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil())
    reading = make_reader()()
    assert isinstance(reading, ResourceReading)
    assert reading.power_w == pytest.approx(157.0)  # mW -> W
    assert reading.vram_mb == pytest.approx(6_700.0)  # bytes -> MB
    assert reading.rss_mb == pytest.approx(21_300.0)
    assert reading.sys_used_mb == pytest.approx(30_100.0)
    assert reading.sys_avail_mb == pytest.approx(1_900.0)


def test_reader_honors_gpu_index(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}
    fake = fake_pynvml()
    fake.nvmlDeviceGetHandleByIndex = lambda i: seen.setdefault("idx", i)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil())
    make_reader(gpu_index=0)
    assert seen["idx"] == 0
