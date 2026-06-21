"""Tests for the break-even figure (the five cost lines vs volume)."""

from __future__ import annotations

from airllm_bench.economics.breakeven import Line
from airllm_bench.plotting.economics_fig import breakeven_fig


def test_breakeven_fig_draws_one_line_per_cost_option() -> None:
    lines = {
        "api": Line(name="API", fixed_usd=0.0, marginal_usd_per_token=1e-6),
        "onprem": Line(name="On-Prem", fixed_usd=1200.0, marginal_usd_per_token=4e-6),
    }
    fig = breakeven_fig(lines, [1e6, 1e7, 1e8])
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None
