"""baseline_hf runner — FP16 transformers on GPU, expected to OOM clean (D3, ADR 0001).

Loads the 32B model in FP16 with ``device_map={"": 0}`` + ``low_cpu_mem_usage=True``
(NOT ``.to("cuda")``, which would stage ~64 GB through 32 GB host RAM -> pagefile).
On the 12 GB card this raises ``OutOfMemoryError`` at load; the harness captures that
as a clean ``ok=False`` result — the capacity-wall data point. ``torch``/``transformers``
import lazily so the module stays keyless in CI; the unit tests inject fakes.

Constructor-injected ``repo_id``/``device`` (the ``Runner`` Protocol's ``load(cfg)`` carries
only per-scenario knobs); the matrix orchestrator builds the runner from ``setup.json``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from airllm_bench.runners import hf_decode
from airllm_bench.runners.protocol import LogitsResult, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig


class BaselineHFRunner:
    """transformers FP16 baseline. ``load`` is expected to raise on VRAM OOM."""

    name = "baseline_hf"

    def __init__(self, repo_id: str, *, device: int = 0) -> None:
        self._repo_id = repo_id
        self._device = device
        self._model: Any = None
        self._tok: Any = None

    def load(self, cfg: ExperimentConfig) -> None:
        """Load FP16 GPU-only; raises ``OutOfMemoryError`` on the 12 GB card (expected).

        ``cfg`` is part of the ``Runner`` contract; the baseline is always FP16 so
        ``cfg.quant`` is intentionally unused here.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self._repo_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._repo_id,
            torch_dtype=torch.float16,
            device_map={"": self._device},
            low_cpu_mem_usage=True,
        )

    def _forward(self, input_ids: Any) -> Any:  # noqa: ANN401 — torch tensors in/out
        return self._model(input_ids).logits

    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]:
        """Greedy temperature-0 decode, one token per forward (D4)."""
        return hf_decode.greedy_stream(
            self._forward, self._tok, prompt, max_new_tokens, self._device
        )

    def logits(self, text: str) -> LogitsResult:
        """One teacher-forced forward pass over ``text`` (D10)."""
        return hf_decode.forward_logits(self._forward, self._tok, text, self._device)

    def unload(self) -> None:
        """Drop references; per-scenario subprocess isolation (T5.4) frees VRAM on exit."""
        self._model = None
        self._tok = None
