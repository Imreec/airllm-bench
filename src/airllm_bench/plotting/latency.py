"""Latency figures: the prefill curve, ITL series, and the cold/warm headline.

Pure functions of prepared data (the result selection lives in ``build``), so
each is trivially testable. TTFT-vs-length shows prefill is streaming-bound for
AirLLM; the ITL series exposes the first-token spike; the cold/warm bars are the
project thesis (page-cache speedup collapsing as the model outgrows RAM).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure

from airllm_bench.plotting.base import new_figure


def ttft_vs_length_fig(curve: Mapping[int, float]) -> Figure:
    """TTFT vs prompt length (points sorted by length) — the prefill curve."""
    fig, ax = new_figure("TTFT vs prompt length", "Prompt tokens", "TTFT (s)")
    lengths = sorted(curve)
    ax.plot(lengths, [curve[n] for n in lengths], marker="o")
    return fig


def itl_fig(series_by_label: Mapping[str, Sequence[float]]) -> Figure:
    """Inter-token latency across the generation, one line per scenario."""
    fig, ax = new_figure("Inter-token latency", "Token index", "ITL (s)")
    for label, series in series_by_label.items():
        ax.plot(range(1, len(series) + 1), list(series), marker=".", label=label)
    if series_by_label:
        ax.legend()
    return fig


def cold_warm_ttft_fig(ttft_by_quant: Mapping[str, tuple[float, float]]) -> Figure:
    """Grouped cold-vs-warm TTFT bars per quant level (the memory-hierarchy result)."""
    fig, ax = new_figure("Cold vs warm TTFT by precision", "Quantization", "TTFT (s)")
    quants = list(ttft_by_quant)
    xs = range(len(quants))
    width = 0.4
    ax.bar([x - width / 2 for x in xs], [ttft_by_quant[q][0] for q in quants], width, label="cold")
    ax.bar([x + width / 2 for x in xs], [ttft_by_quant[q][1] for q in quants], width, label="warm")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(quants)
    ax.legend()
    return fig
