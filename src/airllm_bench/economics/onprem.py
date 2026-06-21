"""On-prem unit cost: amortized CAPEX + measured-energy OPEX (D6).

Returns a fully-amortized ``$/Mtok`` directly comparable to the API price. The
CAPEX is spread over the *utilized* token capacity of its amortization window,
so ``utilization`` is the dominant knob (an idle owned box never amortizes —
the cautionary AirLLM line is exactly this case). Energy cost is driven by the
*measured* GPU energy plus the estimated CPU term (see ``energy``).
"""

from __future__ import annotations

#: Unit conversions (not configuration — pure math, mirrors ``costing._TOKENS_PER_MTOK``).
SECONDS_PER_YEAR = 365 * 24 * 3600
J_PER_KWH = 3_600_000
_TOKENS_PER_MTOK = 1_000_000


def energy_cost_usd(energy_j: float, tariff_usd_per_kwh: float) -> float:
    """USD cost of ``energy_j`` joules at the given tariff."""
    return energy_j / J_PER_KWH * tariff_usd_per_kwh


def capex_per_token_usd(
    capex_usd: float,
    amort_years: float,
    utilization: float,
    throughput_tok_s: float,
) -> float:
    """CAPEX spread over the tokens the box can serve while utilized over its life.

    An idle box (``utilization`` or ``throughput`` of 0) serves zero tokens, so its
    per-token CAPEX is infinite — returned as such rather than dividing by zero.
    """
    capacity_tokens = throughput_tok_s * amort_years * SECONDS_PER_YEAR * utilization
    if capacity_tokens == 0:
        return float("inf")
    return capex_usd / capacity_tokens


def cost_onprem_per_mtok(
    throughput_tok_s: float,
    energy_per_token_j: float,
    tariff_usd_per_kwh: float,
    capex_usd: float,
    amort_years: float,
    utilization: float,
) -> float:
    """Fully-amortized on-prem cost per million tokens (CAPEX + energy OPEX)."""
    capex_tok = capex_per_token_usd(capex_usd, amort_years, utilization, throughput_tok_s)
    opex_tok = energy_cost_usd(energy_per_token_j, tariff_usd_per_kwh)
    return (capex_tok + opex_tok) * _TOKENS_PER_MTOK
