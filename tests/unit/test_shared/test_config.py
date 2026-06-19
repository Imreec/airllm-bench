"""Tests for the versioned config loader and the setup/experiment models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from airllm_bench.shared.config import ConfigVersionError
from airllm_bench.shared.config_models import ExperimentConfig, SetupConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    p = tmp_path / "c.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_loads_and_preserves_raw_data(tmp_path: Path) -> None:
    p = _write(tmp_path, {"version": "1.00", "runner": "mock", "quant": "none", "prompt": "hi"})
    cfg = ExperimentConfig.from_file(p)
    assert cfg.runner == "mock"
    assert cfg.phase == "cold"  # default
    assert cfg.raw_data["version"] == "1.00"


def test_bad_version_prefix_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, {"version": "2.00", "runner": "mock", "quant": "none", "prompt": "hi"})
    with pytest.raises(ConfigVersionError):
        ExperimentConfig.from_file(p)


def test_missing_version_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, {"runner": "mock", "quant": "none", "prompt": "hi"})
    with pytest.raises(ConfigVersionError):
        ExperimentConfig.from_file(p)


def test_real_setup_config_loads() -> None:
    cfg = SetupConfig.from_file(CONFIG_DIR / "setup.json")
    assert cfg.model.repo_id == "Qwen/Qwen2.5-32B-Instruct"
    assert cfg.hardware.vram_gb == 12
    assert cfg.paths.hf_home.startswith("D:")  # disk policy: HF cache on the HDD


def test_real_smoke_experiment_loads() -> None:
    cfg = ExperimentConfig.from_file(CONFIG_DIR / "experiments" / "smoke.json")
    assert cfg.runner == "mock"
    assert cfg.max_new_tokens == 4
