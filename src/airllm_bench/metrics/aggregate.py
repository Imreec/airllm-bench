"""Cross-result aggregation for the report figures (pure, from RunResults)."""

from __future__ import annotations

from collections.abc import Sequence

from airllm_bench.harness.result import RunResult


def ttft_vs_length(results: Sequence[RunResult]) -> dict[int, float]:
    """{prompt_tokens: ttft_s} for results that produced a first token.

    The prefill curve: TTFT rising with input length is the empirical proof that
    Prefill is compute-bound (D4). Later prompt lengths overwrite earlier on a tie.
    """
    return {r.prompt_tokens: r.ttft_s for r in results if r.ttft_s is not None}


def cold_warm_ratio(cold: RunResult, warm: RunResult) -> float | None:
    """cold TPOT / warm TPOT — > 1 means the page cache helped (D4).

    Returns None if either run lacks a TPOT (e.g. a baseline OOM).
    """
    if not cold.tpot_s or not warm.tpot_s:
        return None
    return cold.tpot_s / warm.tpot_s
