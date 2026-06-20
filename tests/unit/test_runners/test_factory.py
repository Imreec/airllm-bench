"""Keyless tests for the runner factory (no heavy backend imported)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from airllm_bench.runners.airllm import AirLLMRunner
from airllm_bench.runners.baseline_hf import BaselineHFRunner
from airllm_bench.runners.factory import build_runner
from airllm_bench.runners.llamacpp import LlamaCppRunner
from airllm_bench.runners.mock import MockRunner
from airllm_bench.shared.config_models import SetupConfig


def _setup(write_config: Callable[..., Path], **runners: Any) -> SetupConfig:
    data = {
        "version": "1.00",
        "hardware": {
            "cpu": "x",
            "cores": 1,
            "gpu": "g",
            "vram_gb": 12,
            "ram_gb": 32,
            "nvme_path": "C:/s",
        },
        "model": {"repo_id": "Qwen/Qwen2.5-32B-Instruct", "params": 32_000_000_000},
        "paths": {"layer_shards_saving_path": "C:/airllm_shards"},
        "runners": runners,
    }
    return SetupConfig.from_file(write_config(data))


def test_mock(write_config: Callable[..., Path]) -> None:
    assert isinstance(build_runner("mock", _setup(write_config)), MockRunner)


def test_baseline_hf_injects_device(write_config: Callable[..., Path]) -> None:
    runner = build_runner("baseline_hf", _setup(write_config, baseline_hf={"device": 0}))
    assert isinstance(runner, BaselineHFRunner)


def test_airllm_injects_shards_path(write_config: Callable[..., Path]) -> None:
    runner = build_runner("airllm", _setup(write_config))
    assert isinstance(runner, AirLLMRunner)
    assert runner._shards_path == "C:/airllm_shards"


def test_llamacpp_injects_gguf_and_offload(write_config: Callable[..., Path]) -> None:
    setup = _setup(write_config, llamacpp={"gguf_path": "m.gguf", "n_gpu_layers": 20})
    runner = build_runner("llamacpp", setup)
    assert isinstance(runner, LlamaCppRunner)
    assert runner._gguf_path == "m.gguf"
    assert runner._n_gpu_layers == 20


def test_llamacpp_without_gguf_fails_loud(write_config: Callable[..., Path]) -> None:
    with pytest.raises(ValueError, match="gguf_path"):
        build_runner("llamacpp", _setup(write_config))


def test_unknown_runner_fails_loud(write_config: Callable[..., Path]) -> None:
    with pytest.raises(ValueError, match="unknown runner"):
        build_runner("bogus", _setup(write_config))
