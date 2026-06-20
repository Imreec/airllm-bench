"""Keyless tests for the shared HF greedy-decode + forward-logits helpers."""

from __future__ import annotations

import sys

import pytest

from airllm_bench.runners import hf_decode
from airllm_bench.runners.protocol import LogitsResult, TokenEvent

from ._fakes import FakeHFModel, FakeTokenizer, fake_torch


def _forward(model: FakeHFModel):  # noqa: ANN202 — local closure factory
    return lambda ids: model(ids).logits


def test_greedy_stream_yields_requested_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    events = list(hf_decode.greedy_stream(_forward(FakeHFModel()), FakeTokenizer(), "hi", 3, 0))
    assert len(events) == 3
    assert all(isinstance(e, TokenEvent) for e in events)
    assert all(isinstance(e.text, str) and e.text for e in events)


def test_greedy_stream_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    a = [e.token_id for e in hf_decode.greedy_stream(_forward(FakeHFModel()), FakeTokenizer(), "x", 2, 0)]
    b = [e.token_id for e in hf_decode.greedy_stream(_forward(FakeHFModel()), FakeTokenizer(), "x", 2, 0)]
    assert a == b


def test_forward_logits_shapes_match() -> None:
    res = hf_decode.forward_logits(_forward(FakeHFModel()), FakeTokenizer(), "abc", 0)
    assert isinstance(res, LogitsResult)
    assert len(res.token_ids) == len(res.logits)  # one logits row per input token
    assert len(res.token_ids) >= 2  # enough for a perplexity pair
