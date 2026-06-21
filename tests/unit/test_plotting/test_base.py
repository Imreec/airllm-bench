"""Tests for the plotting base (headless backend + figure saving)."""

from __future__ import annotations

from pathlib import Path

from airllm_bench.plotting.base import new_figure, save_figure


def test_backend_is_headless_agg() -> None:
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


def test_save_figure_writes_a_nonempty_png(tmp_path: Path) -> None:
    fig, ax = new_figure("t", "x", "y")
    ax.plot([0, 1], [0, 1])
    out = save_figure(fig, tmp_path / "f.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_new_figure_sets_titles_and_labels() -> None:
    _, ax = new_figure("Title", "X label", "Y label")
    assert ax.get_title() == "Title"
    assert ax.get_xlabel() == "X label"
    assert ax.get_ylabel() == "Y label"
