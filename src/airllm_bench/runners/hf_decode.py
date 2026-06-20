"""Shared greedy-decode + forward-logits for the HF-family runners (DRY, PLAN §3).

``baseline_hf`` and ``airllm`` differ only in *where* the logits tensor comes from
(HF: ``model(ids).logits``; AirLLM: ``model(ids)[0]`` — the forward tuple, ADR 0001),
so both pass a ``forward`` callable here. Decoding is greedy / temperature-0 (D4),
one token per forward, yielding a real per-token stream the harness can timestamp.
``torch`` is imported lazily so the module loads keyless in CI; tests inject a fake.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from airllm_bench.runners.protocol import LogitsResult, TokenEvent

Forward = Callable[[Any], Any]  # input_ids -> logits tensor of shape (1, seq, vocab)


def greedy_stream(
    forward: Forward,
    tokenizer: Any,  # noqa: ANN401 — runtime HF tokenizer; kept loose so CI needs no transformers
    prompt: str,
    max_new_tokens: int,
    device: Any,  # noqa: ANN401 — torch device / index
) -> Iterator[TokenEvent]:
    """Greedy-decode ``max_new_tokens`` tokens, yielding one ``TokenEvent`` each."""
    import torch

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(forward(input_ids)[0, -1]))
        yield TokenEvent(token_id=next_id, text=tokenizer.decode([next_id]))
        nxt = torch.tensor([[next_id]], device=device)
        input_ids = torch.cat([input_ids, nxt], dim=1)


def forward_logits(
    forward: Forward,
    tokenizer: Any,  # noqa: ANN401
    text: str,
    device: Any,  # noqa: ANN401
) -> LogitsResult:
    """One teacher-forced forward pass over ``text`` -> ``LogitsResult`` (D10)."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    logits = forward(input_ids)
    return LogitsResult(token_ids=input_ids[0].tolist(), logits=logits[0].tolist())
