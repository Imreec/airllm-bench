"""Typed pydantic models for the HW5 config files (setup + experiment).

Sections allow extra keys so new knobs can land without a schema change. The
economics config model lives in ``economics/`` where it's consumed (Phase 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from airllm_bench.shared.config import load_versioned


class _Section(BaseModel):
    """Base for config sections: keep unknown keys."""

    model_config = ConfigDict(extra="allow")


class HardwareConfig(_Section):
    """The documented machine — read by the report and the cost math (one source)."""

    cpu: str
    cores: int
    gpu: str
    vram_gb: float
    ram_gb: float
    nvme_path: str
    hdd_path: str = ""


class ModelConfig(_Section):
    """The model under test (HF repo id + parameter count)."""

    repo_id: str
    params: int
    format: str = "safetensors"


class PathsConfig(_Section):
    """Where shards, the HF cache, results, and figures live (disk policy, ADR 0001)."""

    layer_shards_saving_path: str
    hf_home: str = ""
    results_dir: str = "results"
    figures_dir: str = "figures"


class SetupConfig(BaseModel):
    """``config/setup.json`` — hardware, model, and paths."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    version: str
    hardware: HardwareConfig
    model: ModelConfig
    paths: PathsConfig
    raw_data: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> SetupConfig:
        """Load and validate ``config/setup.json``."""
        return load_versioned(path, cls)


class ExperimentConfig(BaseModel):
    """``config/experiments/<id>.json`` — exactly ONE scenario (PLAN §3/§4).

    One prompt, one ``phase`` (cold|warm). The length x cold/warm matrix is
    expanded by the Phase-5 orchestrator, not encoded here.
    """

    model_config = ConfigDict(extra="ignore")

    version: str
    runner: str  # baseline_hf | airllm | llamacpp | mock
    quant: str  # none | int8 | nf4 | q4_k_m | q8_0
    prompt: str
    max_new_tokens: int = 8
    phase: str = "cold"  # cold | warm
    reps: int = 1
    raw_data: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate a single-scenario experiment config."""
        return load_versioned(path, cls)
