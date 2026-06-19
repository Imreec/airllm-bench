"""Background resource sampling: GPU power/VRAM + RAM on one clock (PLAN §4, D5).

The pure ``ResourceAggregator`` (peaks + trapezoid-integrated GPU energy) is what
carries the logic and is fully unit-tested; ``ResourceSampler`` is a thin thread
around it that reads a ``reader()`` callable, so tests inject a fake reader + clock
and CI never needs NVML or a GPU. Validated on hardware in the G-SPIKE.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceReading:
    """One instantaneous read of GPU + memory state."""

    power_w: float
    vram_mb: float
    rss_mb: float
    sys_used_mb: float
    sys_avail_mb: float


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """A timestamped reading."""

    t: float
    reading: ResourceReading


class ResourceAggregator:
    """Accumulates samples into peaks + trapezoid-integrated GPU energy. Pure."""

    def __init__(self) -> None:
        self._samples: list[ResourceSample] = []

    def add(self, sample: ResourceSample) -> None:
        """Record one timestamped reading."""
        self._samples.append(sample)

    @property
    def count(self) -> int:
        """Number of samples collected."""
        return len(self._samples)

    def report(self) -> dict[str, float]:
        """Peaks + integrated energy; empty dict when fewer than one sample."""
        s = self._samples
        if not s:
            return {}
        energy = sum(
            (s[i].reading.power_w + s[i - 1].reading.power_w) / 2 * (s[i].t - s[i - 1].t)
            for i in range(1, len(s))
        )
        return {
            "peak_gpu_power_w": max(x.reading.power_w for x in s),
            "gpu_energy_j": energy,
            "peak_vram_mb": max(x.reading.vram_mb for x in s),
            "peak_rss_mb": max(x.reading.rss_mb for x in s),
            "peak_sys_used_mb": max(x.reading.sys_used_mb for x in s),
            "min_sys_avail_mb": min(x.reading.sys_avail_mb for x in s),
            "samples": float(len(s)),
        }


class ResourceSampler:
    """Thread that samples ``reader`` every ``interval_s`` into a ``ResourceAggregator``."""

    def __init__(
        self,
        reader: Callable[[], ResourceReading],
        *,
        interval_s: float = 0.05,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._reader = reader
        self._interval = interval_s
        self._clock = clock
        self._agg = ResourceAggregator()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._agg.add(ResourceSample(t=self._clock(), reading=self._reader()))
            time.sleep(self._interval)

    def __enter__(self) -> ResourceSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def report(self) -> dict[str, float]:
        """Peaks + integrated energy gathered so far."""
        return self._agg.report()
