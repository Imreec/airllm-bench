"""Tests for on-prem unit cost (amortized CAPEX + measured-energy OPEX, D6)."""

from __future__ import annotations

import pytest

from airllm_bench.economics.onprem import (
    SECONDS_PER_YEAR,
    capex_per_token_usd,
    cost_onprem_per_mtok,
    energy_cost_usd,
)


def test_energy_cost_converts_joules_to_usd_via_kwh() -> None:
    # 3.6e6 J = 1 kWh; at 0.2 $/kWh that is $0.20.
    assert energy_cost_usd(3_600_000.0, 0.2) == pytest.approx(0.2)


def test_capex_per_token_amortizes_over_utilized_capacity() -> None:
    cost = capex_per_token_usd(1200.0, amort_years=3.0, utilization=0.5, throughput_tok_s=2.0)
    capacity = 2.0 * 3.0 * SECONDS_PER_YEAR * 0.5
    assert cost == pytest.approx(1200.0 / capacity)


def test_lower_utilization_raises_unit_cost() -> None:
    busy = cost_onprem_per_mtok(2.0, 10.0, 0.2, 1200.0, 3.0, 0.9)
    idle = cost_onprem_per_mtok(2.0, 10.0, 0.2, 1200.0, 3.0, 0.1)
    assert idle > busy


def test_cost_onprem_is_capex_plus_opex_per_mtok() -> None:
    capex_tok = capex_per_token_usd(1200.0, 3.0, 0.5, 2.0)
    opex_tok = energy_cost_usd(10.0, 0.2)
    expected = (capex_tok + opex_tok) * 1_000_000
    assert cost_onprem_per_mtok(2.0, 10.0, 0.2, 1200.0, 3.0, 0.5) == pytest.approx(expected)


def test_zero_utilization_is_infinite_not_a_crash() -> None:
    # An idle box (duty_cycle=0) amortizes CAPEX over zero tokens -> infinite, not ZeroDivisionError.
    assert capex_per_token_usd(1200.0, 3.0, 0.0, 2.0) == float("inf")
    assert cost_onprem_per_mtok(2.0, 10.0, 0.2, 1200.0, 3.0, 0.0) == float("inf")


def test_zero_throughput_is_infinite() -> None:
    assert capex_per_token_usd(1200.0, 3.0, 0.5, 0.0) == float("inf")
