"""airllm runner — layer-streamed inference at none/int8/nf4 (T5.2, ADR 0001).

AirLLM loads the 32B model layer-by-layer from compression-specific shards on the
NVMe, so it fits the 12 GB card by streaming each layer in turn. The compression
level is chosen per scenario from ``cfg.quant``. Two G-SPIKE-specific quirks: AirLLM's
``check_space`` needs the shards dir to exist *before* load, and its forward returns a
TUPLE — logits live at ``out[0]``. Decoding/perplexity reuse the shared ``hf_decode``
core (DRY with ``baseline_hf``; the two differ only in this forward and the load path).

``airllm`` imports lazily so the module stays keyless in CI; the unit tests inject a fake.
Constructor-injected ``repo_id``/``shards_path`` come from ``setup.json`` via the orchestrator.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from airllm_bench.runners import hf_decode
from airllm_bench.runners.protocol import LogitsResult, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig

# cfg.quant -> AirLLM's ``compression`` argument (None = full FP16, no compression).
_COMPRESSION: dict[str, str | None] = {"none": None, "int8": "8bit", "nf4": "4bit"}


class AirLLMRunner:
    """AirLLM AutoModel backend; compression selected per scenario from ``cfg.quant``."""

    name = "airllm"

    def __init__(self, repo_id: str, shards_path: str, *, device: str = "cuda") -> None:
        self._repo_id = repo_id
        self._shards_path = shards_path
        self._device = device
        self._model: Any = None

    def load(self, cfg: ExperimentConfig) -> None:
        """Load via AirLLM AutoModel at the scenario's compression level.

        Raises:
            KeyError: if ``cfg.quant`` is not one of none/int8/nf4 (fail loud on bad config).
        """
        from airllm import AutoModel

        compression = _COMPRESSION[cfg.quant]
        # AirLLM's check_space requires the shards dir to exist first (G-SPIKE finding).
        os.makedirs(self._shards_path, exist_ok=True)
        self._model = AutoModel.from_pretrained(
            self._repo_id,
            compression=compression,
            layer_shards_saving_path=self._shards_path,
        )

    def _forward(self, input_ids: Any) -> Any:  # noqa: ANN401 — torch tensors in/out
        # AirLLM's forward returns a tuple; logits are out[0] (ADR 0001).
        return self._model(input_ids)[0]

    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]:
        """Greedy temperature-0 decode; each token is one full layer-streamed pass (D4)."""
        return hf_decode.greedy_stream(
            self._forward, self._model.tokenizer, prompt, max_new_tokens, self._device
        )

    def logits(self, text: str) -> LogitsResult:
        """One teacher-forced forward pass -> perplexity inputs (D10)."""
        return hf_decode.forward_logits(self._forward, self._model.tokenizer, text, self._device)

    def unload(self) -> None:
        """Drop the model reference; per-scenario subprocess isolation (T5.4) frees the rest."""
        self._model = None
