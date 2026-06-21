"""Tests for the hierarchical roofline figure (ceilings + operating points)."""

from __future__ import annotations

from airllm_bench.plotting.roofline_fig import roofline_fig
from airllm_bench.roofline.points import OperatingPoint
from airllm_bench.shared.config_models import RooflineConfig


def _rc() -> RooflineConfig:
    return RooflineConfig(
        compute_fp16_tflops=68.2,
        hbm_gb_s=912,
        pcie_gb_s=31.5,
        nvme_gb_s=7.0,
        bytes_per_weight={"nf4": 0.5},
    )


def _pt(exp: str, intensity: float, achieved: float) -> OperatingPoint:
    return OperatingPoint(
        exp_id=exp,
        quant="nf4",
        phase="decode",
        flops=1.0,
        bytes_moved=1.0,
        intensity_flops_per_byte=intensity,
        achieved_flops=achieved,
        achieved_bytes_s=1.0,
        binding_tier="pcie",
        tier_bandwidth_bytes_s=31.5e9,
        tier_utilization=0.1,
    )


def test_roofline_has_compute_roof_and_three_diagonals() -> None:
    fig = roofline_fig(_rc(), [_pt("a", 4.0, 1e10)])
    ax = fig.axes[0]
    assert len(ax.lines) == 4  # HBM + PCIe + NVMe diagonals + the compute ceiling
    assert ax.get_legend() is not None


def test_roofline_scatters_the_operating_points() -> None:
    fig = roofline_fig(_rc(), [_pt("a", 4.0, 1e10), _pt("b", 8.0, 2e10)])
    assert len(fig.axes[0].collections) >= 1


def test_roofline_handles_no_points() -> None:
    fig = roofline_fig(_rc(), [])
    assert len(fig.axes[0].lines) == 4
