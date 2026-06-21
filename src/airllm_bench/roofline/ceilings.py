"""Roofline ceilings: the silicon compute roof + memory-tier diagonals (D7).

The classic roofline says attainable FLOP/s = min(compute_peak, bandwidth x
arithmetic_intensity). The hierarchical extension keeps one compute roof but
swaps the bandwidth for the tier the bytes actually traverse — HBM (resident),
PCIe (page cache / RAM), or NVMe (cold disk stream). All ceilings come from the
documented box's ``roofline`` config block (no hardcoded hardware numbers).
"""

from __future__ import annotations

from airllm_bench.shared.config_models import RooflineConfig

#: Unit conversions (pure math, not configuration).
_TFLOPS_TO_FLOPS = 1e12
_GB_TO_BYTES = 1e9  # bandwidth specs are quoted in decimal GB/s

#: Memory tier name -> the ``RooflineConfig`` field holding its GB/s rating.
_TIER_ATTR = {"hbm": "hbm_gb_s", "pcie": "pcie_gb_s", "nvme": "nvme_gb_s"}


def compute_ceiling_flops(rc: RooflineConfig) -> float:
    """The FP16 dense-tensor compute roof in FLOP/s."""
    return rc.compute_fp16_tflops * _TFLOPS_TO_FLOPS


def tier_bandwidth_bytes_s(rc: RooflineConfig, tier: str) -> float:
    """Bandwidth of a memory tier (``hbm`` | ``pcie`` | ``nvme``) in bytes/s."""
    gb_s: float = getattr(rc, _TIER_ATTR[tier])
    return gb_s * _GB_TO_BYTES


def attainable_flops(compute_flops: float, bandwidth_bytes_s: float, intensity: float) -> float:
    """Roofline ceiling at an arithmetic intensity: min(compute, bandwidth x intensity)."""
    return min(compute_flops, bandwidth_bytes_s * intensity)
