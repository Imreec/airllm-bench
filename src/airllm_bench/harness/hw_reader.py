"""The hardware telemetry reader the ResourceSampler thread polls (T5.4, D5).

Returns a closure that reads one instantaneous GPU+memory snapshot: NVML GPU power
and VRAM used (the 3080 Ti), plus psutil process RSS and system used/available.
``pynvml``/``psutil`` are core deps, but ``nvmlInit`` needs a real GPU — so they
import lazily and the unit tests inject fakes via ``sys.modules``, keeping CI keyless.
Windows exposes no system *cached* figure (G-SPIKE), so the page-cache signal is
available-memory + the cold/warm timing contrast, not a cache number (revises D5).
"""

from __future__ import annotations

from collections.abc import Callable

from airllm_bench.harness.sampler import ResourceReading

_MB = 1024 * 1024


def make_reader(*, gpu_index: int = 0) -> Callable[[], ResourceReading]:
    """Initialize NVML + psutil and return a one-shot snapshot reader."""
    import psutil
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    proc = psutil.Process()

    def read() -> ResourceReading:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vm = psutil.virtual_memory()
        return ResourceReading(
            power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,  # mW -> W
            vram_mb=mem.used / _MB,
            rss_mb=proc.memory_info().rss / _MB,
            sys_used_mb=vm.used / _MB,
            sys_avail_mb=vm.available / _MB,
        )

    return read
