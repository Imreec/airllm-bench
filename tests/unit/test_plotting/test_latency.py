"""Tests for the latency figures (TTFT-vs-length, ITL series, cold/warm)."""

from __future__ import annotations

from airllm_bench.plotting.latency import (
    cold_warm_ttft_fig,
    itl_fig,
    ttft_vs_length_fig,
)


def test_ttft_vs_length_plots_sorted_points() -> None:
    fig = ttft_vs_length_fig({256: 3.0, 40: 1.0, 64: 1.5})
    ax = fig.axes[0]
    line = ax.lines[0]
    assert list(line.get_xdata()) == [40, 64, 256]
    assert list(line.get_ydata()) == [1.0, 1.5, 3.0]


def test_itl_fig_draws_one_line_per_series() -> None:
    fig = itl_fig({"nf4 cold": [3.0, 1.0, 1.0], "nf4 warm": [1.0, 1.0]})
    assert len(fig.axes[0].lines) == 2


def test_cold_warm_fig_has_a_cold_and_a_warm_bar_group() -> None:
    fig = cold_warm_ttft_fig({"4-bit": (43.7, 19.3), "8-bit": (54.3, 50.8)})
    ax = fig.axes[0]
    assert len(ax.containers) == 2  # one bar series for cold, one for warm
    assert ax.get_legend() is not None
