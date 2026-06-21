"""Plotting base: a headless backend + a new-figure/save helper (DRY).

The Agg backend is forced before ``pyplot`` is imported so every figure renders
offline with no display — ``make figures`` regenerates the committed PNGs in CI
(D11). All figure builders go through ``new_figure``/``save_figure`` so styling
and the close-after-save discipline live in one place.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend selection
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_DPI = 120


def new_figure(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Axes]:
    """A single-axes figure with the title/labels set (one styling place)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Write ``fig`` to ``path`` as a tight-bbox PNG and close it; return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
