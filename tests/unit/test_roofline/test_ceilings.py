"""Tests for the roofline ceilings (compute roof + memory-tier diagonals)."""

from __future__ import annotations

import pytest

from airllm_bench.roofline.ceilings import (
    attainable_flops,
    compute_ceiling_flops,
    tier_bandwidth_bytes_s,
)
from airllm_bench.shared.config_models import RooflineConfig


def _rc() -> RooflineConfig:
    return RooflineConfig(
        compute_fp16_tflops=68.2,
        hbm_gb_s=912,
        pcie_gb_s=31.5,
        nvme_gb_s=7.0,
        bytes_per_weight={"nf4": 0.5},
    )


def test_compute_ceiling_is_tflops_in_flops() -> None:
    assert compute_ceiling_flops(_rc()) == pytest.approx(68.2e12)


def test_tier_bandwidth_converts_gb_to_bytes() -> None:
    rc = _rc()
    assert tier_bandwidth_bytes_s(rc, "hbm") == pytest.approx(912e9)
    assert tier_bandwidth_bytes_s(rc, "pcie") == pytest.approx(31.5e9)
    assert tier_bandwidth_bytes_s(rc, "nvme") == pytest.approx(7e9)


def test_attainable_is_bandwidth_bound_below_the_ridge() -> None:
    # NVMe 7 GB/s at intensity 4 -> 28 GFLOP/s, far under the 68.2 TFLOP/s compute roof.
    assert attainable_flops(68.2e12, 7e9, 4.0) == pytest.approx(28e9)


def test_attainable_is_compute_bound_above_the_ridge() -> None:
    # HBM at a very high intensity saturates compute, capped at the roof.
    assert attainable_flops(68.2e12, 912e9, 1000.0) == pytest.approx(68.2e12)
