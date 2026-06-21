"""The hierarchical roofline figure (D7) — the report's unifying chart.

One compute ceiling (horizontal) plus the HBM/PCIe/NVMe bandwidth diagonals;
each measured run is scattered at (arithmetic intensity, achieved FLOP/s). The
AirLLM points sit far below the silicon roof on the NVMe/PCIe diagonals, and
nf4-warm sits a tier above nf4-cold — the memory hierarchy made visible.
"""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.figure import Figure

from airllm_bench.plotting.base import new_figure
from airllm_bench.roofline.ceilings import compute_ceiling_flops, tier_bandwidth_bytes_s
from airllm_bench.roofline.points import OperatingPoint
from airllm_bench.shared.config_models import RooflineConfig

_TIERS = ("hbm", "pcie", "nvme")


def _intensity_range(points: Sequence[OperatingPoint]) -> list[float]:
    intensities = [p.intensity_flops_per_byte for p in points] or [1.0]
    return [min(intensities) / 10, max(intensities) * 10]


def roofline_fig(rc: RooflineConfig, points: Sequence[OperatingPoint]) -> Figure:
    """Plot the compute roof + memory diagonals and scatter the operating points."""
    fig, ax = new_figure(
        "Hierarchical roofline", "Arithmetic intensity (FLOP/byte)", "Attainable FLOP/s"
    )
    xs = _intensity_range(points)
    compute = compute_ceiling_flops(rc)
    for tier in _TIERS:
        bw = tier_bandwidth_bytes_s(rc, tier)
        ax.plot(xs, [bw * x for x in xs], label=f"{tier.upper()} {bw / 1e9:.0f} GB/s")
    ax.axhline(
        compute, linestyle="--", color="black", label=f"compute {compute / 1e12:.0f} TFLOP/s"
    )
    if points:
        ax.scatter(
            [p.intensity_flops_per_byte for p in points],
            [p.achieved_flops for p in points],
            zorder=3,
        )
        for p in points:
            # exp_id alone collides: a resident run has both a prefill and a decode
            # point — the phase disambiguates the compute-bound vs memory-bound dot.
            ax.annotate(
                f"{p.exp_id} ({p.phase})",
                (p.intensity_flops_per_byte, p.achieved_flops),
                fontsize=6,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return fig
