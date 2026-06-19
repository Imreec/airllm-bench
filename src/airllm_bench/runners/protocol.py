"""The load-bearing runner contract (PLAN §3).

Every backend (baseline_hf / airllm / llamacpp / mock) implements ``Runner`` so the
harness measures them identically. The runner yields token *content*; the harness
stamps receipt timestamps on one clock (TTFT/TPOT are never total ÷ tokens).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from airllm_bench.shared.config_models import ExperimentConfig


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One generated token's content, as yielded by a runner."""

    token_id: int
    text: str


@dataclass(frozen=True, slots=True)
class LogitsResult:
    """One teacher-forced forward pass for perplexity (PLAN §3, D10).

    ``logits`` is a 2-D float array ``[seq, vocab]`` (torch/numpy on hardware,
    list-of-lists from the mock). Each runner tokenizes natively, so ``token_ids``
    records what *this* runtime used — cross-runtime perplexity needs them to match.
    """

    token_ids: list[int]
    logits: Any  # noqa: ANN401 — array-like; kept loose so the contract needs no torch/numpy


@runtime_checkable
class Runner(Protocol):
    """A measurable inference backend. ``load`` may raise (the baseline OOMs)."""

    name: str

    def load(self, cfg: ExperimentConfig) -> None: ...

    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]: ...

    def logits(self, text: str) -> LogitsResult: ...

    def unload(self) -> None: ...
