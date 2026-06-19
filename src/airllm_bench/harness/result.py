"""``RunResult`` — the single source of truth every analysis module reads (PLAN §4).

One scenario → one ``RunResult`` (one prompt-length, one phase). Emitted as one JSONL
line; ``metrics``/``economics``/``roofline`` consume only this and never touch a model.
Field names mirror the G-SPIKE (ADR 0001): on Windows there is no system *cached*
figure, so memory is RSS + system-used + min-available, not a cache number.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class RunResult(BaseModel):
    """A single measured (or cleanly-failed) scenario."""

    schema_version: str = SCHEMA_VERSION
    exp_id: str
    runner: str
    quant: str
    model: str
    param_count: int

    prompt_tokens: int
    max_new_tokens: int
    phase: str  # cold | warm

    ok: bool  # False on a clean baseline OOM
    error: str | None = None

    # Timing — derived from per-token receipt timestamps (None when ok is False).
    ttft_s: float | None = None
    itl_s: list[float] = Field(default_factory=list)  # full inter-token series, not just the mean
    tpot_s: float | None = None
    throughput_tok_s: float | None = None

    # Resources (peaks over the run; None when not sampled).
    peak_vram_mb: float | None = None
    peak_rss_mb: float | None = None
    peak_sys_used_mb: float | None = None
    min_sys_avail_mb: float | None = None
    gpu_energy_j: float | None = None  # measured via NVML integration
    cpu_energy_j_est: float | None = None  # TDP-based estimate (declared)
    runtime_s: float | None = None

    perplexity: float | None = None  # None on the baseline (no tokens)

    env: dict[str, str] = Field(default_factory=dict)
