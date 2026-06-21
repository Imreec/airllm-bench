"""Structural eval — the break-even lines are well-formed over committed data (D11).

Keyless, deterministic, runs in CI. Builds the five lines from the *committed*
``config/economics.json`` and the *committed* ``results/*.jsonl`` and asserts the
report-level invariants: every cumulative-cost curve is monotonic non-decreasing
in volume, and the cautionary AirLLM on-prem line never costs less than the
realistic on-prem line at any volume (same CAPEX, heavier marginal — D6 "show
with numbers it never amortizes"). (PLAN §7 "break-even monotonic".)

Note: with the committed Israel-tariff constants the on-prem *energy alone*
exceeds the API price, so on-prem never breaks even (a finding, not an invariant)
— we therefore assert structure that holds regardless of the constants.
"""

from __future__ import annotations

from pathlib import Path

from airllm_bench.economics.breakeven import build_lines, cumulative_cost, curve
from airllm_bench.economics.config import EconomicsConfig
from airllm_bench.metrics.load import index_by_exp_id, load_results

REPO_ROOT = Path(__file__).resolve().parents[3]
_VOLUMES = [0, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]


def _lines() -> dict[str, object]:
    cfg = EconomicsConfig.from_file(REPO_ROOT / "config" / "economics.json")
    idx = index_by_exp_id(load_results(REPO_ROOT / "results"))
    return build_lines(
        cfg,
        model_key="qwen2.5-32b",
        realistic=idx["llamacpp-q4-warm"],
        airllm=idx["airllm-nf4-warm"],
        input_per_output_ratio=1.0,
    )


def test_every_breakeven_curve_is_monotonic() -> None:
    for name, line in _lines().items():
        ys = curve(line, _VOLUMES)  # type: ignore[arg-type]
        assert ys == sorted(ys), name


def test_cautionary_airllm_line_dominates_the_realistic_line() -> None:
    lines = _lines()
    realistic, airllm = lines["onprem_realistic"], lines["onprem_airllm"]
    for v in _VOLUMES:
        assert cumulative_cost(airllm, v) >= cumulative_cost(realistic, v), v  # type: ignore[arg-type]
