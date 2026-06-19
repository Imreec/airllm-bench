"""Tests for the resource aggregator (logic) and the sampler thread (smoke)."""

from __future__ import annotations

import time

import pytest

from airllm_bench.harness.sampler import (
    ResourceAggregator,
    ResourceReading,
    ResourceSample,
    ResourceSampler,
)


def _reading(power: float = 100.0, vram: float = 1000.0, avail: float = 20000.0) -> ResourceReading:
    return ResourceReading(
        power_w=power, vram_mb=vram, rss_mb=500.0, sys_used_mb=8000.0, sys_avail_mb=avail
    )


def test_aggregator_empty_reports_nothing() -> None:
    assert ResourceAggregator().report() == {}


def test_aggregator_peaks_and_trapezoid_energy() -> None:
    agg = ResourceAggregator()
    agg.add(ResourceSample(t=0.0, reading=_reading(power=100, vram=1000, avail=20000)))
    agg.add(ResourceSample(t=1.0, reading=_reading(power=200, vram=3000, avail=15000)))
    agg.add(ResourceSample(t=2.0, reading=_reading(power=100, vram=2000, avail=18000)))
    rep = agg.report()
    assert rep["peak_gpu_power_w"] == 200.0
    assert rep["peak_vram_mb"] == 3000.0
    assert rep["min_sys_avail_mb"] == 15000.0
    # (100+200)/2*1 + (200+100)/2*1 = 300
    assert rep["gpu_energy_j"] == pytest.approx(300.0)
    assert rep["samples"] == 3.0


def test_sampler_thread_collects_samples() -> None:
    powers = iter([100.0, 200.0, 300.0, 400.0])
    clocks = iter([0.0, 0.1, 0.2, 0.3, 0.4])

    def reader() -> ResourceReading:
        return _reading(power=next(powers, 500.0))

    def clock() -> float:
        return next(clocks, 9.0)

    with ResourceSampler(reader, interval_s=0.001, clock=clock) as sampler:
        time.sleep(0.05)
    rep = sampler.report()
    assert rep.get("samples", 0) >= 1
    assert "gpu_energy_j" in rep
