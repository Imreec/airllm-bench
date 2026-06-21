"""Tests for the throughput and perplexity bar figures."""

from __future__ import annotations

from airllm_bench.plotting.throughput_quality import perplexity_fig, throughput_fig


def test_throughput_fig_has_one_bar_per_scenario() -> None:
    fig = throughput_fig({"nf4": 0.05, "int8": 0.02, "fp16": 0.009})
    assert len(fig.axes[0].patches) == 3


def test_throughput_xtick_labels_are_rotated_so_they_dont_overlap() -> None:
    # The x labels are long exp_ids and there are many of them — horizontal text
    # collides into an unreadable smear. They must be angled.
    fig = throughput_fig(
        {"airllm-int8-cold": 0.02, "airllm-int8-warm": 0.02, "llamacpp-q4-warm": 0.99}
    )
    rotations = [label.get_rotation() for label in fig.axes[0].get_xticklabels()]
    assert rotations
    assert all(r != 0 for r in rotations)


def test_perplexity_fig_has_one_bar_per_quant() -> None:
    fig = perplexity_fig({"4-bit": 39.9, "8-bit": 24.2, "fp16": 23.7})
    assert len(fig.axes[0].patches) == 3
