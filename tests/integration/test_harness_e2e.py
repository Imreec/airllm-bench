"""End-to-end harness runs against the mock runner (keyless): OOM + sampler mapping."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from airllm_bench.harness.run import run
from airllm_bench.runners.mock import MockRunner
from airllm_bench.shared.config_models import ExperimentConfig


def _seq_clock() -> Callable[[], float]:
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += 1.0
        return state["t"]

    return clock


class _FakeSampler:
    def __init__(self, report: dict[str, float]) -> None:
        self._report = report

    def __enter__(self) -> _FakeSampler:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def report(self) -> dict[str, float]:
        return self._report


def _exp(data: dict[str, Any], write_config: Callable[..., Path]) -> ExperimentConfig:
    return ExperimentConfig.from_file(write_config(data))


def test_clean_oom_is_captured_not_raised(
    experiment_config_data: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    exp = _exp(experiment_config_data, write_config)
    result = run(
        MockRunner(load_error="CUDA out of memory"),
        exp,
        exp_id="oom",
        model="m",
        param_count=1,
        prompt_tokens=0,
        clock=_seq_clock(),
    )
    assert not result.ok
    assert "out of memory" in (result.error or "")
    assert result.ttft_s is None
    assert result.runtime_s is not None


def test_sampler_report_maps_into_result(
    experiment_config_data: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    exp = _exp(experiment_config_data, write_config)
    report = {
        "peak_vram_mb": 6700.0,
        "gpu_energy_j": 150.0,
        "peak_rss_mb": 4000.0,
        "peak_sys_used_mb": 20000.0,
        "min_sys_avail_mb": 16000.0,
    }
    result = run(
        MockRunner(),
        exp,
        exp_id="s",
        model="m",
        param_count=1,
        prompt_tokens=5,
        sampler=_FakeSampler(report),
        clock=_seq_clock(),
    )
    assert result.ok
    assert result.peak_vram_mb == 6700.0
    assert result.gpu_energy_j == 150.0
    assert result.min_sys_avail_mb == 16000.0
