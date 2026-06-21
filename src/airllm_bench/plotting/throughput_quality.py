"""Throughput and quality (perplexity) bar figures.

Speed-vs-bits and quality-vs-bits — together they frame the quantization
trade-off the report turns into an economics/roofline argument.
"""

from __future__ import annotations

from collections.abc import Mapping

from matplotlib.figure import Figure

from airllm_bench.plotting.base import new_figure


def throughput_fig(tput_by_label: Mapping[str, float]) -> Figure:
    """Decode throughput (tok/s) per scenario — speed falls as precision rises."""
    fig, ax = new_figure("Decode throughput", "Scenario", "Tokens / s")
    labels = list(tput_by_label)
    ax.bar(labels, [tput_by_label[k] for k in labels])
    # The scenario exp_ids are long and numerous — angle them so they don't collide
    # into an unreadable smear above the axis. (set_xticks first to pin the locator.)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    return fig


def perplexity_fig(ppl_by_quant: Mapping[str, float]) -> Figure:
    """Perplexity per quant level — lower is better (8-bit near-lossless)."""
    fig, ax = new_figure("Perplexity by precision", "Quantization", "Perplexity")
    labels = list(ppl_by_quant)
    ax.bar(labels, [ppl_by_quant[k] for k in labels])
    return fig
