"""Keyless fakes for the HF-family runner unit tests (no torch, no GPU, no model).

Just enough tensor/tokenizer/model semantics to drive the real greedy-decode and
forward-logits loops in ``hf_decode`` and the runners. Injected via
``monkeypatch.setitem(sys.modules, "torch", fake_torch())`` so the lazy
``import torch`` inside runner methods resolves to these instead of the real wheel.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

VOCAB = 5


class FakeTensor:
    """A nested-list tensor supporting the few ops the decode loops use."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def to(self, device: Any) -> FakeTensor:  # noqa: ANN401 — mirrors torch's loose API
        return self

    def tolist(self) -> Any:
        return self.data

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        cur = self.data
        for k in key if isinstance(key, tuple) else (key,):
            cur = cur[k]
        return FakeTensor(cur) if isinstance(cur, list) else cur


def _argmax(t: Any) -> int:  # noqa: ANN401
    row = t.data if isinstance(t, FakeTensor) else t
    return row.index(max(row))


def _cat(tensors: list[FakeTensor], dim: int = 1) -> FakeTensor:
    a, b = tensors[0].data, tensors[1].data
    return FakeTensor([ra + rb for ra, rb in zip(a, b, strict=True)])


def fake_torch(*, cuda_available: bool = True) -> ModuleType:
    """A stand-in ``torch`` module exposing only what the runners touch."""
    mod = ModuleType("torch")
    mod.argmax = _argmax  # type: ignore[attr-defined]
    mod.tensor = lambda data, device=None: FakeTensor(data)  # type: ignore[attr-defined]
    mod.cat = _cat  # type: ignore[attr-defined]
    mod.float16 = "float16"  # type: ignore[attr-defined]
    mod.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda_available, empty_cache=lambda: None
    )
    return mod


class FakeTokenizer:
    """Maps text -> a few token ids and back, deterministically."""

    def __call__(self, text: str, return_tensors: str | None = None) -> SimpleNamespace:
        ids = [ord(c) % VOCAB for c in text][:3] or [1]
        return SimpleNamespace(input_ids=FakeTensor([ids]))

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + (i % VOCAB)) for i in ids)


def _logits_for(seq_len: int) -> FakeTensor:
    """A deterministic (1, seq, VOCAB) logits tensor."""
    return FakeTensor([[[float((i + j) % VOCAB) for j in range(VOCAB)] for i in range(seq_len)]])


class FakeHFModel:
    """A transformers-style model: ``model(ids).logits`` (1, seq, vocab)."""

    def __call__(self, input_ids: FakeTensor) -> SimpleNamespace:
        return SimpleNamespace(logits=_logits_for(len(input_ids.data[0])))


class FakeAirLLMModel:
    """An AirLLM-style model: forward returns a TUPLE, logits at ``out[0]`` (ADR 0001)."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def __call__(self, input_ids: FakeTensor) -> tuple[FakeTensor]:
        return (_logits_for(len(input_ids.data[0])),)


def fake_transformers(*, oom: bool = False) -> ModuleType:
    """A stand-in ``transformers`` module with the two Auto classes the baseline uses."""
    mod = ModuleType("transformers")

    def _model_from_pretrained(repo_id: str, **kwargs: Any) -> FakeHFModel:
        if oom:
            msg = "CUDA out of memory"
            raise RuntimeError(msg)
        return FakeHFModel()

    mod.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda repo_id, **kw: FakeTokenizer()
    )
    mod.AutoModelForCausalLM = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=_model_from_pretrained
    )
    return mod
