"""Structural eval — roofline operating points are correct from inputs (D11/D7).

Keyless, deterministic, runs in CI. Builds decode + prefill operating points from
the *committed* ``config/setup.json`` roofline block + the *committed*
``results/*.jsonl`` and asserts: the arithmetic reproduces from (param_count,
bytes_per_weight, throughput/ttft); each path lands on the physically-correct
tier (resident -> HBM, AirLLM cold -> NVMe, AirLLM warm climbs to PCIe only when
it fits usable RAM); prefill is higher-intensity than decode (the compute-bound
point D7 requires); and no point exceeds its binding bandwidth (utilization <= 1,
a hard physical bound). (PLAN §7 "roofline points correct from inputs".)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from airllm_bench.metrics.load import index_by_exp_id, load_results
from airllm_bench.roofline.points import (
    operating_point,
    prefill_operating_point,
    resident_runner_names,
)
from airllm_bench.shared.config_models import SetupConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def _setup() -> SetupConfig:
    return SetupConfig.from_file(REPO_ROOT / "config" / "setup.json")


def _points() -> dict[str, object]:
    setup = _setup()
    assert setup.roofline is not None
    resident = resident_runner_names(setup.runners)
    idx = index_by_exp_id(load_results(REPO_ROOT / "results"))
    return {
        e: operating_point(r, setup.roofline, setup.hardware.ram_gb, resident)
        for e, r in idx.items()
    }


def test_nf4_intensity_reproduces_from_inputs() -> None:
    pt = _points()["airllm-nf4-warm"]
    assert pt.intensity_flops_per_byte == pytest.approx(4.0)  # 2 / 0.5  # type: ignore[union-attr]
    assert pt.flops == pytest.approx(2 * 32_000_000_000)  # type: ignore[union-attr]


def test_each_path_binds_on_its_physical_tier() -> None:
    pts = _points()
    assert pts["llamacpp-q4-warm"].binding_tier == "hbm"  # type: ignore[union-attr]
    assert pts["airllm-nf4-cold"].binding_tier == "nvme"  # type: ignore[union-attr]
    assert pts["airllm-nf4-warm"].binding_tier == "pcie"  # fits 32 GB  # type: ignore[union-attr]
    assert (
        pts["airllm-none-warm"].binding_tier == "nvme"
    )  # 64 GB, won't fit  # type: ignore[union-attr]


def test_prefill_is_higher_intensity_than_decode() -> None:
    setup = _setup()
    assert setup.roofline is not None
    resident = resident_runner_names(setup.runners)
    idx = index_by_exp_id(load_results(REPO_ROOT / "results"))
    r = idx["llamacpp-q4-warm"]
    decode = operating_point(r, setup.roofline, setup.hardware.ram_gb, resident)
    prefill = prefill_operating_point(r, setup.roofline, setup.hardware.ram_gb, resident)
    assert decode is not None
    assert prefill is not None
    # Prefill does prompt_tokens passes per weight read -> strictly more compute-bound.
    assert prefill.intensity_flops_per_byte > decode.intensity_flops_per_byte
    assert prefill.achieved_flops > decode.achieved_flops


def test_no_point_exceeds_its_binding_bandwidth() -> None:
    for exp_id, pt in _points().items():
        if pt is None:  # the baseline OOM never decoded
            continue
        assert pt.tier_utilization <= 1.0, exp_id  # type: ignore[union-attr]
