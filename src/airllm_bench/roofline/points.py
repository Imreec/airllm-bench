"""Operating-point placement on the hierarchical roofline (D7).

Each decode step reads every weight once, so per token: FLOPs = 2 x params,
bytes = params x bytes_per_weight, and arithmetic intensity = 2 / bytes_per_weight
(independent of model size). The binding tier follows the *physical data path*:
resident runtimes (llama.cpp, baseline GPU) read from HBM; AirLLM streams cold
weights from NVMe; a warm AirLLM run whose footprint fits RAM is served from the
page cache over PCIe — the upward shift of the same tool as quantization pulls
the working set under RAM (the memory-hierarchy result). Failed runs (no
throughput) have no operating point.
"""

from __future__ import annotations

from pydantic import BaseModel

from airllm_bench.harness.result import RunResult
from airllm_bench.roofline.ceilings import _GB_TO_BYTES, tier_bandwidth_bytes_s
from airllm_bench.shared.config_models import RooflineConfig

_FLOPS_PER_PARAM_PER_TOKEN = 2  # one multiply-add per weight per token
_RESIDENT_RUNNERS = ("baseline_hf", "llamacpp")


class OperatingPoint(BaseModel):
    """One run placed on the roofline, with the arithmetic shown (D7)."""

    exp_id: str
    quant: str
    flops_per_token: float
    bytes_per_token: float
    intensity_flops_per_byte: float
    achieved_flops: float
    achieved_bytes_s: float
    binding_tier: str  # hbm | pcie | nvme
    tier_bandwidth_bytes_s: float
    tier_utilization: float  # achieved / tier ceiling


def fits_in_ram(param_count: int, bytes_per_weight: float, ram_gb: float) -> bool:
    """Whether the quantized weights fit the OS page cache (RAM)."""
    return param_count * bytes_per_weight < ram_gb * _GB_TO_BYTES


def binding_tier(result: RunResult, fits_cache: bool) -> str:
    """The memory tier the bytes actually traverse for this run."""
    if result.runner in _RESIDENT_RUNNERS:
        return "hbm"
    if result.phase == "cold":
        return "nvme"
    return "pcie" if fits_cache else "nvme"


def operating_point(
    result: RunResult,
    rc: RooflineConfig,
    ram_gb: float,
) -> OperatingPoint | None:
    """Place one ``RunResult`` on the roofline, or None if it never decoded."""
    if not result.throughput_tok_s:
        return None
    bpw = rc.bytes_per_weight[result.quant]
    flops_tok = _FLOPS_PER_PARAM_PER_TOKEN * result.param_count
    bytes_tok = result.param_count * bpw
    tier = binding_tier(result, fits_in_ram(result.param_count, bpw, ram_gb))
    tier_bw = tier_bandwidth_bytes_s(rc, tier)
    achieved_bytes_s = bytes_tok * result.throughput_tok_s
    return OperatingPoint(
        exp_id=result.exp_id,
        quant=result.quant,
        flops_per_token=flops_tok,
        bytes_per_token=bytes_tok,
        intensity_flops_per_byte=flops_tok / bytes_tok,
        achieved_flops=flops_tok * result.throughput_tok_s,
        achieved_bytes_s=achieved_bytes_s,
        binding_tier=tier,
        tier_bandwidth_bytes_s=tier_bw,
        tier_utilization=achieved_bytes_s / tier_bw,
    )
