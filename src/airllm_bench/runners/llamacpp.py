"""llamacpp runner — the realistic GGUF competitor with partial GPU offload (T5.3, D3).

llama.cpp runs a quantized GGUF (Q4_K_M / Q8) with ``n_gpu_layers`` offloaded to the
3080 Ti and the rest on CPU — the practical "what would you actually deploy" baseline
the report contrasts against AirLLM's layer streaming. Unlike the HF family this backend
exposes real token ids token-by-token (``Llama.generate``), so the stream stays honest
without HF tokenizer assumptions; perplexity uses its own ``logits_all`` eval (D10) —
each runtime tokenizes natively, never sharing input ids (PLAN §3).

``llama_cpp`` imports lazily so the module stays keyless in CI; the unit tests inject a fake.
``gguf_path``/``n_gpu_layers`` are constructor-injected from ``setup.json`` by the orchestrator.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from airllm_bench.runners.protocol import LogitsResult, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig


class LlamaCppRunner:
    """llama-cpp-python GGUF backend; quant is fixed by the GGUF file itself."""

    name = "llamacpp"

    def __init__(self, gguf_path: str, *, n_gpu_layers: int = 0) -> None:
        self._gguf_path = gguf_path
        self._n_gpu_layers = n_gpu_layers
        self._llm: Any = None

    def load(self, cfg: ExperimentConfig) -> None:  # cfg: contract-required; quant set by the GGUF
        """Construct the Llama context; ``logits_all`` is on so perplexity can read logits."""
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self._gguf_path,
            n_gpu_layers=self._n_gpu_layers,
            logits_all=True,
            verbose=False,
        )

    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]:
        """Greedy (temp=0) token-by-token generation; stop at EOS or the token budget (D4)."""
        tokens = self._llm.tokenize(prompt.encode("utf-8"))
        for count, token in enumerate(self._llm.generate(tokens, temp=0.0)):
            if count >= max_new_tokens or token == self._llm.token_eos():
                break
            text = self._llm.detokenize([token]).decode("utf-8", errors="replace")
            yield TokenEvent(token_id=token, text=text)

    def logits(self, text: str) -> LogitsResult:
        """One teacher-forced eval over ``text``; native tokenization (PLAN §3, D10)."""
        tokens = self._llm.tokenize(text.encode("utf-8"))
        self._llm.reset()
        self._llm.eval(tokens)
        rows = [list(row) for row in self._llm.eval_logits]
        return LogitsResult(token_ids=tokens, logits=rows)

    def unload(self) -> None:
        """Drop the context; per-scenario subprocess isolation (T5.4) frees VRAM on exit."""
        self._llm = None
