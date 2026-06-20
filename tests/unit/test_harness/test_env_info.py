"""Keyless tests for environment metadata capture."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from airllm_bench.harness.env_info import collect_env


def _fake_torch(version: str = "2.6.0+cu124") -> ModuleType:
    mod = ModuleType("torch")
    mod.__version__ = version  # type: ignore[attr-defined]
    mod.version = SimpleNamespace(cuda="12.4")  # type: ignore[attr-defined]
    mod.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    return mod


def _fake_pynvml(version: bytes | str = b"555.99") -> ModuleType:
    mod = ModuleType("pynvml")
    mod.nvmlInit = lambda: None  # type: ignore[attr-defined]
    mod.nvmlSystemGetDriverVersion = lambda: version  # type: ignore[attr-defined]
    return mod


def test_collect_env_blank_when_stack_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """os/python always present; torch+driver blank when the stack is absent.

    Forced deterministically (``sys.modules[name] = None`` makes ``import`` raise) so the
    result is identical on the keyless CI box and the dev box that *does* have a GPU.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    env = collect_env()
    assert env["os"]
    assert env["python"]
    assert env["torch"] == ""
    assert env["cuda_available"] == ""
    assert env["driver"] == ""


def test_collect_env_captures_torch_and_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    env = collect_env()
    assert env["torch"] == "2.6.0+cu124"
    assert env["cuda"] == "12.4"
    assert env["cuda_available"] == "True"
    assert env["driver"] == "555.99"  # bytes decoded


def test_driver_version_accepts_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(version="560.1"))
    env = collect_env()
    assert env["driver"] == "560.1"
