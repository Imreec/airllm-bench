"""Timing derivation from per-token receipt timestamps (PLAN §4, D4).

TTFT and TPOT come from the *gaps between timestamps*, never total ÷ tokens —
that split is what lets the report demonstrate Prefill vs Decode. The full ITL
series is kept (AirLLM's is expected bimodal: page-cache hit vs miss).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Timing:
    """Derived timing for one scenario."""

    ttft_s: float | None
    itl_s: list[float] = field(default_factory=list)
    tpot_s: float | None = None
    throughput_tok_s: float | None = None


def compute_timing(t_gen_start: float, token_times: list[float]) -> Timing:
    """Derive TTFT/ITL/TPOT/throughput from token receipt timestamps.

    ``t_gen_start`` is when generation began (after load); ``token_times`` are the
    clock readings as each token arrived. Returns an empty ``Timing`` if no tokens.
    """
    if not token_times:
        return Timing(ttft_s=None)
    ttft = token_times[0] - t_gen_start
    itl = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    tpot = sum(itl) / len(itl) if itl else None
    total = token_times[-1] - t_gen_start
    throughput = len(token_times) / total if total > 0 else None
    return Timing(ttft_s=ttft, itl_s=itl, tpot_s=tpot, throughput_tok_s=throughput)


def itl_stats(itl_s: list[float]) -> dict[str, float]:
    """Mean + nearest-rank percentiles of the inter-token series (empty dict if none)."""
    if not itl_s:
        return {}
    ordered = sorted(itl_s)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * len(ordered)), len(ordered) - 1)
        return ordered[idx]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "min": ordered[0],
        "max": ordered[-1],
    }
