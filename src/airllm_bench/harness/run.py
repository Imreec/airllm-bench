"""The one measurement harness: exactly one scenario -> one RunResult (PLAN §4).

Drives a runner under a resource sampler on one clock, deriving TTFT/TPOT from
per-token receipt timestamps. A clean load-time OOM (the baseline) is captured
into ``ok=False`` rather than crashing the pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from airllm_bench.harness.result import RunResult
from airllm_bench.metrics.quality import perplexity
from airllm_bench.metrics.timing import compute_timing
from airllm_bench.runners.protocol import Runner
from airllm_bench.shared.config_models import ExperimentConfig


@runtime_checkable
class Sampler(Protocol):
    """A resource sampler usable as a context manager that reports peaks/energy."""

    def __enter__(self) -> object: ...
    def __exit__(self, *args: object) -> None: ...
    def report(self) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class _Core:
    ok: bool
    error: str | None
    runtime_s: float
    ttft_s: float | None = None
    itl_s: list[float] = field(default_factory=list)
    tpot_s: float | None = None
    throughput_tok_s: float | None = None
    perplexity: float | None = None


def _safe_perplexity(runner: Runner, prompt: str) -> float | None:
    """Best-effort perplexity; None if the runner can't yield usable logits."""
    try:
        lg = runner.logits(prompt)
        return perplexity(lg.logits, lg.token_ids)
    except Exception:  # noqa: BLE001 — perplexity is optional; never fail the run for it
        return None


def _measure(runner: Runner, exp: ExperimentConfig, clock: Callable[[], float]) -> _Core:
    t0 = clock()
    try:
        runner.load(exp)
    except Exception as exc:  # noqa: BLE001 — capture a clean OOM as a result, not a crash
        return _Core(ok=False, error=f"{type(exc).__name__}: {exc}", runtime_s=clock() - t0)
    t_gen = clock()
    token_times = [clock() for _ in runner.stream(exp.prompt, exp.max_new_tokens)]
    timing = compute_timing(t_gen, token_times)
    ppl = _safe_perplexity(runner, exp.prompt)
    runner.unload()
    return _Core(
        ok=True,
        error=None,
        runtime_s=clock() - t0,
        ttft_s=timing.ttft_s,
        itl_s=timing.itl_s,
        tpot_s=timing.tpot_s,
        throughput_tok_s=timing.throughput_tok_s,
        perplexity=ppl,
    )


def run(
    runner: Runner,
    exp: ExperimentConfig,
    *,
    exp_id: str,
    model: str,
    param_count: int,
    prompt_tokens: int,
    sampler: Sampler | None = None,
    clock: Callable[[], float] = time.perf_counter,
    env: dict[str, str] | None = None,
) -> RunResult:
    """Run one scenario and return its RunResult (PLAN §4)."""
    base: dict[str, Any] = {
        "exp_id": exp_id,
        "runner": exp.runner,
        "quant": exp.quant,
        "model": model,
        "param_count": param_count,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": exp.max_new_tokens,
        "phase": exp.phase,
        "env": env or {},
    }
    if sampler is not None:
        with sampler:
            core = _measure(runner, exp, clock)
        report = sampler.report()
    else:
        core = _measure(runner, exp, clock)
        report = {}
    if not core.ok:
        return RunResult(ok=False, error=core.error, runtime_s=core.runtime_s, **base)
    return RunResult(
        ok=True,
        ttft_s=core.ttft_s,
        itl_s=core.itl_s,
        tpot_s=core.tpot_s,
        throughput_tok_s=core.throughput_tok_s,
        runtime_s=core.runtime_s,
        perplexity=core.perplexity,
        peak_vram_mb=report.get("peak_vram_mb"),
        peak_rss_mb=report.get("peak_rss_mb"),
        peak_sys_used_mb=report.get("peak_sys_used_mb"),
        min_sys_avail_mb=report.get("min_sys_avail_mb"),
        gpu_energy_j=report.get("gpu_energy_j"),
        **base,
    )
