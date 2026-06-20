"""Keyless tests for the llamacpp runner (llama_cpp faked in sys.modules)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from airllm_bench.runners.llamacpp import LlamaCppRunner, _ensure_cuda_dll_path
from airllm_bench.runners.protocol import LogitsResult, Runner, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig

from ._fakes import fake_llama_cpp


def test_ensure_cuda_dll_path_is_safe_noop() -> None:
    """No-op off Windows (the CI path); never raises so load() can always call it."""
    assert _ensure_cuda_dll_path() is None


def _cfg(write_config: Callable[..., Path], **over: Any) -> ExperimentConfig:
    data = {
        "version": "1.00",
        "runner": "llamacpp",
        "quant": "q4_k_m",
        "prompt": "What is virtual memory?",
        "max_new_tokens": 3,
        "phase": "warm",
        "reps": 1,
        **over,
    }
    return ExperimentConfig.from_file(write_config(data))


def test_satisfies_runner_protocol() -> None:
    assert isinstance(LlamaCppRunner("model.gguf"), Runner)


def test_load_enables_logits_and_offload(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path]
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    runner = LlamaCppRunner("model.gguf", n_gpu_layers=20)
    runner.load(_cfg(write_config))
    assert runner._llm.logits_all is True  # required for perplexity
    assert runner._llm.n_gpu_layers == 20  # partial GPU offload (the competitor)
    assert runner._llm.model_path == "model.gguf"


def test_stream_respects_token_budget(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path]
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    runner = LlamaCppRunner("model.gguf")
    runner.load(_cfg(write_config))
    events = list(runner.stream("hi", 3))
    assert len(events) == 3
    assert all(isinstance(e, TokenEvent) for e in events)
    assert all(e.token_id >= 0 and e.text for e in events)  # real ids, not placeholders


def test_stream_stops_at_eos(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path]
) -> None:
    """The fake yields 5 ids then EOS; a larger budget must stop at EOS, not overrun."""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    runner = LlamaCppRunner("model.gguf")
    runner.load(_cfg(write_config))
    events = list(runner.stream("hi", 50))
    assert len(events) == 5  # bounded by EOS, not the budget


def test_stream_handles_generator_exhaustion(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path]
) -> None:
    """If generation ends without EOS and under budget, the stream just stops cleanly."""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    runner = LlamaCppRunner("model.gguf")
    runner.load(_cfg(write_config))
    runner._llm.generate = lambda tokens, temp=0.0: iter([1, 2])  # finite, no EOS
    events = list(runner.stream("hi", 50))
    assert len(events) == 2


def test_logits_returns_per_token_rows(
    monkeypatch: pytest.MonkeyPatch, write_config: Callable[..., Path]
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    runner = LlamaCppRunner("model.gguf")
    runner.load(_cfg(write_config))
    res = runner.logits("abc")
    assert isinstance(res, LogitsResult)
    assert len(res.token_ids) == len(res.logits)
    assert len(res.token_ids) >= 2  # enough for a perplexity pair
    runner.unload()
    assert runner._llm is None
