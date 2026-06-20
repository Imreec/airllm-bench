"""Shared fake pynvml/psutil for the hardware-reader + scenario-main tests."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

_MB = 1024 * 1024


def fake_pynvml() -> ModuleType:
    """A pynvml stand-in exposing the calls ``make_reader`` makes."""
    mod = ModuleType("pynvml")
    mod.nvmlInit = lambda: None
    mod.nvmlDeviceGetHandleByIndex = lambda i: f"handle{i}"
    mod.nvmlDeviceGetPowerUsage = lambda h: 157_000  # mW
    mod.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(used=6_700 * _MB)
    return mod


def fake_psutil() -> ModuleType:
    """A psutil stand-in for the process RSS + system memory reads."""
    mod = ModuleType("psutil")
    mod.Process = lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=21_300 * _MB))
    mod.virtual_memory = lambda: SimpleNamespace(used=30_100 * _MB, available=1_900 * _MB)
    return mod
