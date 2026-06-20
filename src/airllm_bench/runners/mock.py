"""A keyless, GPU-free runner for harness wiring and the CI smoke test (D11).

Emits canned tokens and tiny canned logits so the whole harness → RunResult →
metrics path can be exercised with no model, no GPU, no key. Can also simulate a
clean load-time OOM (the baseline scenario) via ``load_error``.
"""

from __future__ import annotations

from collections.abc import Iterator

from airllm_bench.runners.protocol import LogitsResult, TokenEvent
from airllm_bench.shared.config_models import ExperimentConfig

_DEFAULT_TOKENS = ["Vir", "tual", " memory", " pages", "."]


class MockRunner:
    """Implements the ``Runner`` Protocol without any inference backend."""

    name = "mock"

    def __init__(
        self,
        tokens: list[str] | None = None,
        *,
        load_error: str | None = None,
        stream_error: str | None = None,
    ) -> None:
        self._tokens = tokens if tokens is not None else list(_DEFAULT_TOKENS)
        self._load_error = load_error
        self._stream_error = stream_error
        self.loaded = False

    def load(self, cfg: ExperimentConfig) -> None:
        """Mark as loaded, or raise to simulate a clean OOM (baseline scenario)."""
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        self.loaded = True

    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]:
        """Yield up to ``max_new_tokens`` canned tokens, or raise to simulate a mid-gen failure."""
        if self._stream_error is not None:
            raise RuntimeError(self._stream_error)
        for i, text in enumerate(self._tokens[:max_new_tokens]):
            yield TokenEvent(token_id=i, text=text)

    def logits(self, text: str) -> LogitsResult:
        """Return tiny canned logits (3 positions over a vocab of 4)."""
        rows = [[2.0, 1.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [1.0, 0.0, 2.0, 0.0]]
        return LogitsResult(token_ids=[1, 2, 0], logits=rows)

    def unload(self) -> None:
        """Release (no-op for the mock)."""
        self.loaded = False
