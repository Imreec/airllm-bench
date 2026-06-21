"""The break-even figure: cumulative cost vs token volume for the five lines (D6).

Log-log axes because both volume and cost span orders of magnitude. Where an
on-prem line stays above the API line across the whole range, on-prem never
amortizes — the report's economic finding read straight off the chart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure

from airllm_bench.economics.breakeven import Line, curve
from airllm_bench.plotting.base import new_figure


def breakeven_fig(lines: Mapping[str, Line], volumes: Sequence[float]) -> Figure:
    """Plot each cost line's cumulative cost over ``volumes`` (output tokens)."""
    fig, ax = new_figure("Break-even: cost vs volume", "Output tokens", "Cumulative cost (USD)")
    for line in lines.values():
        ax.plot(list(volumes), curve(line, volumes), marker="o", label=line.name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    return fig
