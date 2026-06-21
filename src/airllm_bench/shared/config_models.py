"""Typed pydantic models for the HW5 config files (setup + experiment).

Sections allow extra keys so new knobs can land without a schema change. The
economics config model lives in ``economics/`` where it's consumed (Phase 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


class RooflineConfig(_Section):
    """Hierarchical-roofline ceilings + per-quant byte cost for the box (D7).

    Memory-hierarchy diagonals (GB/s) and the silicon compute roof (TFLOP/s);
    ``bytes_per_weight`` maps each ``quant`` to its on-the-wire byte cost per
    parameter (fp16=2, int8=1, nf4=0.5, q4_k_m≈0.56).
    """

    compute_fp16_tflops: float
    compute_fp16_sparse_tflops: float = 0.0
    hbm_gb_s: float
    pcie_gb_s: float
    nvme_gb_s: float
    bytes_per_weight: dict[str, float]
    # RAM reserved by OS/Python/torch before it can serve as page cache (D7/L-10).
    os_overhead_gb: float = 0.0


class SetupConfig(BaseModel):
    """``config/setup.json`` — hardware, model, and paths."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    version: str
    hardware: HardwareConfig
    model: ModelConfig
    paths: PathsConfig
    # Roofline ceilings (optional: only the Phase-6 roofline tier reads it).
    roofline: RooflineConfig | None = None
    # Per-runner constructor params the factory injects (device / gguf_path / n_gpu_layers).
    # Kept loose (data, not code) so a runtime knob lands without a schema change.
    runners: dict[str, Any] = Field(default_factory=dict)
    # Cold-cache flush: the standby-list tool command + the minimum freed-MB the flush
    # must achieve to count as a genuine cold cache (ADR 0001). Empty = warm-only runs.
    cold_flush: dict[str, Any] = Field(default_factory=dict)
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
    # Native token count, pinned by the matrix sizer when known; the worker falls back
    # to a keyless whitespace estimate when absent (kept honest in the report).
    prompt_tokens: int | None = None
    max_new_tokens: int = 8
    phase: str = "cold"  # cold | warm
    reps: int = 1
    raw_data: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate a single-scenario experiment config."""
        return load_versioned(path, cls)
