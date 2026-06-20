"""Keyless tests for the airllm runner (airllm/torch faked in sys.modules)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from airllm_bench.runners.airllm import AirLLMRunner
from airllm_bench.runners.protocol import LogitsResult, Runner, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig

from ._fakes import fake_airllm, fake_torch


def _cfg(write_config: Callable[..., Path], **over: Any) -> ExperimentConfig:
    data = {
        "version": "1.00",
        "runner": "airllm",
        "quant": "nf4",
        "prompt": "What is virtual memory?",
        "max_new_tokens": 3,
        "phase": "warm",
        "reps": 1,
        **over,
    }
    return ExperimentConfig.from_file(write_config(data))


def test_satisfies_runner_protocol(tmp_path: Path) -> None:
    assert isinstance(AirLLMRunner("Qwen/Qwen2.5-32B-Instruct", str(tmp_path)), Runner)


def test_load_creates_shards_dir_and_passes_compression(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    record: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "airllm", fake_airllm(record))
    shards = tmp_path / "shards"  # does not exist yet — load must create it
    runner = AirLLMRunner("Qwen/Qwen2.5-32B-Instruct", str(shards))
    runner.load(_cfg(write_config, quant="nf4"))
    assert shards.is_dir()  # AirLLM check_space needs it pre-created (G-SPIKE)
    assert record["compression"] == "4bit"
    assert record["layer_shards_saving_path"] == str(shards)


@pytest.mark.parametrize(("quant", "expected"), [("none", None), ("int8", "8bit"), ("nf4", "4bit")])
def test_quant_maps_to_compression(
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[..., Path],
    tmp_path: Path,
    quant: str,
    expected: str | None,
) -> None:
    record: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "airllm", fake_airllm(record))
    runner = AirLLMRunner("repo", str(tmp_path))
    runner.load(_cfg(write_config, quant=quant))
    assert record["compression"] == expected


def test_unknown_quant_fails_loud(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "airllm", fake_airllm())
    runner = AirLLMRunner("repo", str(tmp_path))
    with pytest.raises(KeyError):
        runner.load(_cfg(write_config, quant="bogus"))


def test_stream_and_logits(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "airllm", fake_airllm())
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    runner = AirLLMRunner("repo", str(tmp_path))
    runner.load(_cfg(write_config))
    events = list(runner.stream("hi", 3))
    assert len(events) == 3
    assert all(isinstance(e, TokenEvent) for e in events)
    res = runner.logits("abc")  # AirLLM forward tuple -> out[0] (ADR 0001)
    assert isinstance(res, LogitsResult)
    assert len(res.token_ids) == len(res.logits)
    runner.unload()
