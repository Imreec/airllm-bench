"""Keyless tests for the baseline_hf runner (torch/transformers faked in sys.modules)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from airllm_bench.runners.baseline_hf import BaselineHFRunner
from airllm_bench.runners.protocol import LogitsResult, Runner, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig

from ._fakes import fake_torch, fake_transformers


def _cfg(write_config: Callable[..., Path], **over: Any) -> ExperimentConfig:
    data = {
        "version": "1.00",
        "runner": "baseline_hf",
        "quant": "none",
        "prompt": "What is virtual memory?",
        "max_new_tokens": 3,
        "phase": "cold",
        "reps": 1,
        **over,
    }
    return ExperimentConfig.from_file(write_config(data))


def _patch_stack(monkeypatch: pytest.MonkeyPatch, *, oom: bool = False) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers(oom=oom))


def test_satisfies_runner_protocol() -> None:
    assert isinstance(BaselineHFRunner("Qwen/Qwen2.5-32B-Instruct"), Runner)


def test_load_then_stream_and_logits(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path]
) -> None:
    _patch_stack(monkeypatch)
    runner = BaselineHFRunner("Qwen/Qwen2.5-32B-Instruct")
    runner.load(_cfg(write_config))
    events = list(runner.stream("hi", 3))
    assert len(events) == 3
    assert all(isinstance(e, TokenEvent) for e in events)
    res = runner.logits("abc")
    assert isinstance(res, LogitsResult)
    runner.unload()


def test_load_propagates_clean_oom(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path]
) -> None:
    """The 12 GB card OOMs at FP16 load; load must raise so the harness records ok=False."""
    _patch_stack(monkeypatch, oom=True)
    runner = BaselineHFRunner("Qwen/Qwen2.5-32B-Instruct")
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.load(_cfg(write_config))
