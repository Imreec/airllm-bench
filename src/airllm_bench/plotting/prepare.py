"""Data-prep selectors: turn committed RunResults into figure-ready inputs.

Pure functions (the figure builders stay dumb), so the result-selection rules —
which scenarios feed which figure — are unit-tested in one place. Quant codes are
mapped to the report's display labels here.
"""

from __future__ import annotations

from collections.abc import Sequence

from airllm_bench.harness.result import RunResult
from airllm_bench.metrics.aggregate import ttft_vs_length

#: Internal quant code -> the precision label used in the report figures.
_QUANT_LABEL = {"nf4": "4-bit", "int8": "8-bit", "none": "FP16", "q4_k_m": "Q4_K_M"}


def _select(results: Sequence[RunResult], **pred: object) -> list[RunResult]:
    return [r for r in results if all(getattr(r, k) == v for k, v in pred.items())]


def _first(results: Sequence[RunResult], **pred: object) -> RunResult | None:
    matches = _select(results, **pred)
    return matches[0] if matches else None


def _baseline_length(results: Sequence[RunResult], runner: str, quant: str) -> int | None:
    """Shortest prompt length measured for a runner/quant — the cold/warm baseline.

    Derived (not hardcoded) so the cold and warm queries stay symmetric: a cold
    vs warm delta must compare the *same* prompt length, even if a length sweep
    is later run in either phase.
    """
    lengths = [r.prompt_tokens for r in _select(results, runner=runner, quant=quant)]
    return min(lengths) if lengths else None


def ttft_curve(results: Sequence[RunResult]) -> dict[int, float]:
    """{prompt_tokens: TTFT} for the AirLLM nf4 warm length sweep (prefill curve)."""
    return ttft_vs_length(_select(results, runner="airllm", quant="nf4", phase="warm"))


def cold_warm_ttft(results: Sequence[RunResult]) -> dict[str, tuple[float, float]]:
    """{precision label: (cold TTFT, warm TTFT)} for each AirLLM quant (the thesis)."""
    out: dict[str, tuple[float, float]] = {}
    for quant, label in _QUANT_LABEL.items():
        base = _baseline_length(results, "airllm", quant)
        if base is None:
            continue
        cold = _first(results, runner="airllm", quant=quant, phase="cold", prompt_tokens=base)
        warm = _first(results, runner="airllm", quant=quant, phase="warm", prompt_tokens=base)
        if cold and warm and cold.ttft_s is not None and warm.ttft_s is not None:
            out[label] = (cold.ttft_s, warm.ttft_s)
    return out


def throughput_by_scenario(results: Sequence[RunResult]) -> dict[str, float]:
    """{exp_id: throughput} for every run that produced a decode rate."""
    return {r.exp_id: r.throughput_tok_s for r in results if r.throughput_tok_s is not None}


def perplexity_by_quant(results: Sequence[RunResult]) -> dict[str, float]:
    """{precision label: perplexity}, one warm AirLLM run per quant.

    Pinned to ONE global prompt length shared across quants — not a per-quant
    baseline. This figure compares *across* quants, so all bars must sit at the
    same length: perplexity across prompt lengths is not comparable (longer
    context lowers it). Under an asymmetric matrix a per-quant baseline could
    compare nf4@40 against int8@64; the global pin avoids that, and a quant with
    no run at the shared length is dropped rather than mis-compared.
    """
    lengths = [r.prompt_tokens for r in _select(results, runner="airllm", phase="warm")]
    base = min(lengths) if lengths else None
    out: dict[str, float] = {}
    for quant, label in _QUANT_LABEL.items():
        run = _first(results, runner="airllm", quant=quant, phase="warm", prompt_tokens=base)
        if run and run.perplexity is not None:
            out[label] = run.perplexity
    return out


def realistic_run(results: Sequence[RunResult]) -> RunResult | None:
    """The realistic-deployment run (llama.cpp warm) — the primary on-prem line."""
    return _first(results, runner="llamacpp", phase="warm")


def cautionary_airllm_run(results: Sequence[RunResult]) -> RunResult | None:
    """The AirLLM nf4 warm run at the baseline length — the cautionary on-prem line."""
    base = _baseline_length(results, "airllm", "nf4")
    if base is None:
        return None
    return _first(results, runner="airllm", quant="nf4", phase="warm", prompt_tokens=base)


def itl_series(results: Sequence[RunResult]) -> dict[str, list[float]]:
    """{label: inter-token series} for the nf4 cold vs warm runs (the spike)."""
    out: dict[str, list[float]] = {}
    base = _baseline_length(results, "airllm", "nf4")
    for phase in ("cold", "warm"):
        run = _first(results, runner="airllm", quant="nf4", phase=phase, prompt_tokens=base)
        if run and run.itl_s:
            out[f"nf4 {phase}"] = run.itl_s
    return out
