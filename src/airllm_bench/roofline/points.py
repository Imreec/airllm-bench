"""Operating-point placement on the hierarchical roofline (D7).

Every weight is read once per forward pass, so per token: bytes = params x
bytes_per_weight. **Decode** does one pass per token -> FLOPs = 2 x params,
intensity = 2 / bytes_per_weight (size-independent, memory-bound). **Prefill**
does ``prompt_tokens`` passes before the first token -> FLOPs = 2 x params x
prompt_tokens, intensity = 2 x prompt_tokens / bytes_per_weight (much higher,
compute-bound) — the two points D7 needs to validate the textbook roofline.

The binding tier follows the *physical data path*: resident runtimes (weights in
VRAM, flagged ``resident`` in setup.json) read from HBM; AirLLM streams cold
weights from NVMe; a warm AirLLM run whose footprint fits the *usable* page cache
(RAM minus OS/Python/torch overhead) is served over PCIe. Failed runs (no
throughput / no TTFT) have no operating point.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from pydantic import BaseModel

from airllm_bench.harness.result import RunResult
from airllm_bench.roofline.ceilings import _GB_TO_BYTES, tier_bandwidth_bytes_s
from airllm_bench.shared.config_models import RooflineConfig

_FLOPS_PER_PARAM_PER_TOKEN = 2  # one multiply-add per weight per token


class OperatingPoint(BaseModel):
    """One run/phase placed on the roofline, with the arithmetic shown (D7)."""

    exp_id: str
    quant: str
    phase: str  # prefill | decode
    flops: float  # FLOPs of the work unit (one token for decode; the whole prefill)
    bytes_moved: float  # weight bytes read for that work unit (params x bytes_per_weight)
    intensity_flops_per_byte: float
    achieved_flops: float
    achieved_bytes_s: float
    binding_tier: str  # hbm | pcie | nvme
    tier_bandwidth_bytes_s: float
    tier_utilization: float  # achieved / tier ceiling


def resident_runner_names(runners: dict[str, Any]) -> frozenset[str]:
    """Runner names flagged ``resident`` in setup.json (weights live in VRAM)."""
    return frozenset(name for name, cfg in runners.items() if cfg.get("resident"))


def fits_in_ram(
    param_count: int,
    bytes_per_weight: float,
    ram_gb: float,
    os_overhead_gb: float = 0.0,
) -> bool:
    """Whether the weights fit the *usable* page cache (RAM minus OS overhead)."""
    usable_bytes = (ram_gb - os_overhead_gb) * _GB_TO_BYTES
    return param_count * bytes_per_weight < usable_bytes


def binding_tier(result: RunResult, fits_cache: bool, resident_runners: Collection[str]) -> str:
    """The memory tier the bytes actually traverse for this run."""
    if result.runner in resident_runners:
        return "hbm"
    if result.phase == "cold":
        return "nvme"
    return "pcie" if fits_cache else "nvme"


def _place(
    result: RunResult,
    rc: RooflineConfig,
    ram_gb: float,
    resident_runners: Collection[str],
    phase: str,
    flops: float,
    elapsed_s: float,
) -> OperatingPoint:
    bpw = rc.bytes_per_weight[result.quant]
    bytes_tok = result.param_count * bpw
    tier = binding_tier(
        result, fits_in_ram(result.param_count, bpw, ram_gb, rc.os_overhead_gb), resident_runners
    )
    tier_bw = tier_bandwidth_bytes_s(rc, tier)
    achieved_bytes_s = bytes_tok / elapsed_s
    return OperatingPoint(
        exp_id=result.exp_id,
        quant=result.quant,
        phase=phase,
        flops=flops,
        bytes_moved=bytes_tok,
        intensity_flops_per_byte=flops / bytes_tok,
        achieved_flops=flops / elapsed_s,
        achieved_bytes_s=achieved_bytes_s,
        binding_tier=tier,
        tier_bandwidth_bytes_s=tier_bw,
        tier_utilization=achieved_bytes_s / tier_bw,
    )


def operating_point(
    result: RunResult,
    rc: RooflineConfig,
    ram_gb: float,
    resident_runners: Collection[str],
) -> OperatingPoint | None:
    """The decode point (memory-bound), or None if the run never decoded."""
    if not result.throughput_tok_s:
        return None
    flops = _FLOPS_PER_PARAM_PER_TOKEN * result.param_count
    return _place(
        result, rc, ram_gb, resident_runners, "decode", flops, 1 / result.throughput_tok_s
    )


def prefill_operating_point(
    result: RunResult,
    rc: RooflineConfig,
    ram_gb: float,
    resident_runners: Collection[str],
) -> OperatingPoint | None:
    """The prefill point (compute-bound), or None without a TTFT / prompt."""
    if not result.ttft_s or not result.prompt_tokens:
        return None
    flops = _FLOPS_PER_PARAM_PER_TOKEN * result.param_count * result.prompt_tokens
    return _place(result, rc, ram_gb, resident_runners, "prefill", flops, result.ttft_s)
