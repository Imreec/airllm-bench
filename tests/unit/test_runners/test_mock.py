"""Tests for the mock runner and its conformance to the Runner Protocol."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from airllm_bench.runners.mock import MockRunner
from airllm_bench.runners.protocol import LogitsResult, Runner, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig


def _cfg(data: dict[str, Any], write_config: Callable[..., Path]) -> ExperimentConfig:
    return ExperimentConfig.from_file(write_config(data))


def test_mock_satisfies_runner_protocol() -> None:
    assert isinstance(MockRunner(), Runner)


def test_load_then_unload(
    experiment_config_data: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    runner = MockRunner()
    runner.load(_cfg(experiment_config_data, write_config))
    assert runner.loaded
    runner.unload()
    assert not runner.loaded


def test_load_error_simulates_oom(
    experiment_config_data: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    runner = MockRunner(load_error="CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.load(_cfg(experiment_config_data, write_config))


def test_stream_respects_max_new_tokens() -> None:
    runner = MockRunner(tokens=["a", "b", "c", "d"])
    events = list(runner.stream("hi", 2))
    assert len(events) == 2
    assert all(isinstance(e, TokenEvent) for e in events)
    assert events[0].text == "a"


def test_stream_error_raises_on_iteration() -> None:
    runner = MockRunner(stream_error="kernel died mid-generation")
    with pytest.raises(RuntimeError, match="mid-generation"):
        list(runner.stream("hi", 4))


def test_logits_returns_result() -> None:
    res = MockRunner().logits("hi")
    assert isinstance(res, LogitsResult)
    assert len(res.token_ids) == 3
    assert len(res.logits) == 3
