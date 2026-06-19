"""Shared fixtures for the airllm-bench test suite."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def setup_config_data() -> dict[str, Any]:
    """A complete, valid ``config/setup.json`` payload."""
    return {
        "version": "1.00",
        "hardware": {
            "cpu": "AMD Ryzen 9 5900X",
            "cores": 12,
            "gpu": "NVIDIA GeForce RTX 3080 Ti",
            "vram_gb": 12,
            "ram_gb": 32,
            "nvme_path": "C:/airllm_shards",
            "hdd_path": "D:/",
        },
        "model": {"repo_id": "Qwen/Qwen2.5-32B-Instruct", "params": 32_000_000_000},
        "paths": {"layer_shards_saving_path": "C:/airllm_shards", "hf_home": "D:/hf_cache"},
    }


@pytest.fixture
def experiment_config_data() -> dict[str, Any]:
    """A complete, valid single-scenario experiment payload (mock runner)."""
    return {
        "version": "1.00",
        "runner": "mock",
        "quant": "none",
        "prompt": "What is virtual memory?",
        "max_new_tokens": 4,
        "phase": "cold",
        "reps": 1,
    }


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    """Return a factory that writes a config payload to a temp file and returns its path."""

    def _write(payload: dict[str, Any], name: str = "config.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write
