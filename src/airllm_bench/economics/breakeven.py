"""The five-line break-even model (D6): cumulative cost vs token volume.

Each line is ``fixed_usd + marginal_usd_per_token x volume``. On-prem options
carry the up-front CAPEX as their fixed cost and a tiny marginal (energy only),
so they start expensive and flatten; API/cloud start free and rise — they cross
at a break-even volume. Volume is counted in *output* tokens; input tokens are
implied by ``input_per_output_ratio`` (a representative request shape).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from airllm_bench.economics.config import EconomicsConfig
from airllm_bench.economics.energy import energy_per_token_j
from airllm_bench.economics.onprem import energy_cost_usd
from airllm_bench.harness.result import RunResult

_TOKENS_PER_MTOK = 1_000_000
_SECONDS_PER_HOUR = 3600


class Line(BaseModel):
    """One cost line: a fixed up-front cost plus a per-token marginal."""

    name: str
    fixed_usd: float
    marginal_usd_per_token: float


def cumulative_cost(line: Line, volume_tokens: float) -> float:
    """Total USD to serve ``volume_tokens`` output tokens on ``line``."""
    return line.fixed_usd + line.marginal_usd_per_token * volume_tokens


def curve(line: Line, volumes: Sequence[float]) -> list[float]:
    """Cumulative cost at each volume (non-decreasing — marginal/fixed are >= 0)."""
    return [cumulative_cost(line, v) for v in volumes]


def breakeven_volume(a: Line, b: Line) -> float | None:
    """Output-token volume where lines ``a`` and ``b`` cost the same.

    None when the marginals are equal (parallel lines never cross).
    """
    slope_gap = b.marginal_usd_per_token - a.marginal_usd_per_token
    if slope_gap == 0:
        return None
    volume = (a.fixed_usd - b.fixed_usd) / slope_gap
    # A non-positive crossing means the lines only meet at v<=0: there is no
    # break-even in feasible (positive) volume, so report it as "never".
    return volume if volume > 0 else None


def _onprem_marginal(cfg: EconomicsConfig, result: RunResult) -> float:
    """Energy cost per output token for an on-prem runner (0 if unmeasured)."""
    e = energy_per_token_j(
        result, cfg.hardware_power.cpu_tdp_w, cfg.hardware_power.cpu_load_fraction
    )
    return 0.0 if e is None else energy_cost_usd(e, cfg.electricity.tariff_usd_per_kwh)


def build_lines(
    cfg: EconomicsConfig,
    model_key: str,
    realistic: RunResult,
    airllm: RunResult,
    input_per_output_ratio: float,
) -> dict[str, Line]:
    """Construct the five report lines from config + measured results (D6)."""
    rate = cfg.token_rate(model_key)
    in_tok = rate.input_per_mtok / _TOKENS_PER_MTOK
    out_tok = rate.output_per_mtok / _TOKENS_PER_MTOK
    cache = cfg.caching
    cached_factor = (
        1 - cache.cached_fraction
    ) + cache.cached_fraction * cache.cached_input_discount
    capex = cfg.capex_usd()
    # A stalled/failed run (0 or None throughput) costs infinitely per token, not zero:
    # guard the denominator instead of folding it into a division that would read as free.
    tput = realistic.throughput_tok_s
    cloud_per_tok = (
        float("inf") if not tput else cfg.cloud_gpu.usd_per_hour / _SECONDS_PER_HOUR / tput
    )
    return {
        "api": Line(
            name="API",
            fixed_usd=0.0,
            marginal_usd_per_token=out_tok + input_per_output_ratio * in_tok,
        ),
        "api_cached": Line(
            name="API + caching",
            fixed_usd=0.0,
            marginal_usd_per_token=out_tok + input_per_output_ratio * in_tok * cached_factor,
        ),
        "onprem_realistic": Line(
            name="On-Prem (realistic)",
            fixed_usd=capex,
            marginal_usd_per_token=_onprem_marginal(cfg, realistic),
        ),
        "onprem_airllm": Line(
            name="On-Prem (AirLLM)",
            fixed_usd=capex,
            marginal_usd_per_token=_onprem_marginal(cfg, airllm),
        ),
        "cloud": Line(name="Cloud GPU", fixed_usd=0.0, marginal_usd_per_token=cloud_per_tok),
    }
