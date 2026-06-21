"""The hierarchical roofline figure (D7) — the report's unifying chart.

One compute ceiling (horizontal) plus the HBM/PCIe/NVMe bandwidth diagonals;
each measured run is scattered at (arithmetic intensity, achieved FLOP/s) and
**coloured by the ceiling it binds to**, so a point visually belongs to its
diagonal. The AirLLM points sit far below the silicon roof on the NVMe/PCIe
diagonals, and nf4-warm sits a tier above nf4-cold — the memory hierarchy made
visible. Labels are compact and offset on short leader lines so the low-left
decode cluster stays readable.
"""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from airllm_bench.plotting.base import new_figure
from airllm_bench.roofline.ceilings import compute_ceiling_flops, tier_bandwidth_bytes_s
from airllm_bench.roofline.points import OperatingPoint
from airllm_bench.shared.config_models import RooflineConfig

#: Each tier drawn (and its points coloured) in one consistent colour.
_TIER_COLOR = {"hbm": "#1f77b4", "pcie": "#ff7f0e", "nvme": "#2ca02c"}
#: Staggered label offsets (points) cycled to keep clustered labels from overlapping.
_LABEL_OFFSETS = ((8, 10), (8, -12), (8, 24), (8, -26), (-8, 14), (-8, -18))


def _intensity_range(points: Sequence[OperatingPoint]) -> list[float]:
    intensities = [p.intensity_flops_per_byte for p in points] or [1.0]
    return [min(intensities) / 10, max(intensities) * 10]


def _short_label(point: OperatingPoint) -> str:
    """Drop the runner prefix (quant already identifies it): ``nf4-cold (decode)``."""
    return f"{point.exp_id.split('-', 1)[-1]} ({point.phase})"


def _annotate(ax: Axes, points: Sequence[OperatingPoint]) -> None:
    for i, p in enumerate(points):
        color = _TIER_COLOR.get(p.binding_tier, "black")
        dx, dy = _LABEL_OFFSETS[i % len(_LABEL_OFFSETS)]
        ax.annotate(
            _short_label(p),
            (p.intensity_flops_per_byte, p.achieved_flops),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="right" if dx < 0 else "left",
            fontsize=6,
            color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": color, "alpha": 0.6},
        )


def roofline_fig(rc: RooflineConfig, points: Sequence[OperatingPoint]) -> Figure:
    """Plot the compute roof + memory diagonals and scatter the operating points."""
    fig, ax = new_figure(
        "Hierarchical roofline", "Arithmetic intensity (FLOP/byte)", "Attainable FLOP/s"
    )
    xs = _intensity_range(points)
    for tier, color in _TIER_COLOR.items():
        bw = tier_bandwidth_bytes_s(rc, tier)
        ax.plot(xs, [bw * x for x in xs], color=color, label=f"{tier.upper()} {bw / 1e9:.0f} GB/s")
    compute = compute_ceiling_flops(rc)
    ax.axhline(
        compute, linestyle="--", color="black", label=f"compute {compute / 1e12:.0f} TFLOP/s"
    )
    if points:
        ax.scatter(
            [p.intensity_flops_per_byte for p in points],
            [p.achieved_flops for p in points],
            c=[_TIER_COLOR.get(p.binding_tier, "black") for p in points],
            zorder=3,
        )
        _annotate(ax, points)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return fig
