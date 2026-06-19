"""Structural eval — the harness captures every field a run must carry (D11).

Keyless, deterministic, runs in CI. The invariant: a successful mock run populates
all timing + quality + runtime fields; a clean OOM sets ok=False + error with no
timing. (The eval-harness skill's "harness-captures-all-fields".)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from airllm_bench.harness.run import run
from airllm_bench.runners.mock import MockRunner
from airllm_bench.shared.config_models import ExperimentConfig

_REQUIRED_ON_SUCCESS = ("ttft_s", "tpot_s", "throughput_tok_s", "perplexity", "runtime_s")


def _seq_clock() -> Callable[[], float]:
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += 1.0
        return state["t"]

    return clock


def _exp(data: dict[str, Any], write_config: Callable[..., Path]) -> ExperimentConfig:
    return ExperimentConfig.from_file(write_config(data))


def test_successful_run_populates_all_fields(
    experiment_config_data: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    result = run(
        MockRunner(),
        _exp(experiment_config_data, write_config),
        exp_id="x",
        model="m",
        param_count=1,
        prompt_tokens=5,
        clock=_seq_clock(),
    )
    assert result.ok
    assert result.error is None
    assert result.itl_s  # non-empty series
    for field_name in _REQUIRED_ON_SUCCESS:
        assert getattr(result, field_name) is not None, field_name


def test_oom_run_is_clean_failure(
    experiment_config_data: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    result = run(
        MockRunner(load_error="OutOfMemoryError"),
        _exp(experiment_config_data, write_config),
        exp_id="x",
        model="m",
        param_count=1,
        prompt_tokens=0,
        clock=_seq_clock(),
    )
    assert result.ok is False
    assert result.error is not None
    assert result.ttft_s is None
