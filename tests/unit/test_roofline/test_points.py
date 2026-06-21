"""Tests for operating-point placement on the hierarchical roofline (D7)."""

from __future__ import annotations

from typing import Any

import pytest

from airllm_bench.harness.result import RunResult
from airllm_bench.roofline.points import (
    fits_in_ram,
    operating_point,
    prefill_operating_point,
    resident_runner_names,
)
from airllm_bench.shared.config_models import RooflineConfig

_P = 32_000_000_000
_RESIDENT = frozenset({"baseline_hf", "llamacpp"})


def _rc() -> RooflineConfig:
    return RooflineConfig(
        compute_fp16_tflops=68.2,
        hbm_gb_s=912,
        pcie_gb_s=31.5,
        nvme_gb_s=7.0,
        bytes_per_weight={"none": 2.0, "int8": 1.0, "nf4": 0.5, "q4_k_m": 0.56},
        os_overhead_gb=5.0,
    )


def _r(runner: str, quant: str, phase: str, tput: float | None = 0.05) -> RunResult:
    return RunResult.model_validate(
        {
            "exp_id": f"{runner}-{quant}-{phase}",
            "runner": runner,
            "quant": quant,
            "model": "m",
            "param_count": _P,
            "prompt_tokens": 40,
            "max_new_tokens": 10,
            "phase": phase,
            "ok": tput is not None,
            "throughput_tok_s": tput,
            "ttft_s": 2.0 if tput is not None else None,
        }
    )


def test_resident_runner_names_reads_config_flag() -> None:
    runners: dict[str, Any] = {
        "baseline_hf": {"resident": True},
        "airllm": {"resident": False},
        "llamacpp": {"resident": True},
    }
    assert resident_runner_names(runners) == _RESIDENT


def test_fits_in_ram_subtracts_os_overhead() -> None:
    assert fits_in_ram(_P, 0.5, ram_gb=32, os_overhead_gb=5) is True  # 16 GB < 27 GB
    # A 30 GB footprint fits raw 32 GB but NOT once 5 GB overhead is reserved.
    assert fits_in_ram(60_000_000_000, 0.5, ram_gb=32, os_overhead_gb=5) is False  # 30 GB > 27 GB


def test_decode_point_intensity_and_per_token_arithmetic() -> None:
    pt = operating_point(_r("airllm", "nf4", "warm"), _rc(), ram_gb=32, resident_runners=_RESIDENT)
    assert pt is not None
    assert pt.phase == "decode"
    assert pt.flops == pytest.approx(2 * _P)
    assert pt.bytes_moved == pytest.approx(_P * 0.5)
    assert pt.intensity_flops_per_byte == pytest.approx(4.0)  # 2 / 0.5
    assert pt.achieved_flops == pytest.approx(2 * _P * 0.05)


def test_prefill_point_uses_ttft_and_scales_intensity_with_prompt_length() -> None:
    pt = prefill_operating_point(
        _r("llamacpp", "q4_k_m", "warm"), _rc(), ram_gb=32, resident_runners=_RESIDENT
    )
    assert pt is not None
    assert pt.phase == "prefill"
    # Prefill does prompt_tokens passes over the weights: intensity = 2*tokens/bpw.
    assert pt.intensity_flops_per_byte == pytest.approx(2 * 40 / 0.56)
    assert pt.flops == pytest.approx(2 * _P * 40)
    assert pt.achieved_flops == pytest.approx(2 * _P * 40 / 2.0)  # / ttft_s
    assert pt.binding_tier == "hbm"


def test_prefill_point_none_without_ttft() -> None:
    r = _r("llamacpp", "q4_k_m", "warm")
    r.ttft_s = None
    assert prefill_operating_point(r, _rc(), 32, _RESIDENT) is None


def test_resident_runtimes_bind_on_hbm() -> None:
    assert (
        operating_point(_r("llamacpp", "q4_k_m", "warm"), _rc(), 32, _RESIDENT).binding_tier
        == "hbm"
    )
    assert (
        operating_point(_r("baseline_hf", "none", "cold"), _rc(), 32, _RESIDENT).binding_tier
        == "hbm"
    )


def test_airllm_cold_binds_on_nvme() -> None:
    assert operating_point(_r("airllm", "nf4", "cold"), _rc(), 32, _RESIDENT).binding_tier == "nvme"


def test_airllm_warm_climbs_to_pcie_only_when_it_fits_ram() -> None:
    assert operating_point(_r("airllm", "nf4", "warm"), _rc(), 32, _RESIDENT).binding_tier == "pcie"
    assert (
        operating_point(_r("airllm", "none", "warm"), _rc(), 32, _RESIDENT).binding_tier == "nvme"
    )


def test_failed_run_has_no_operating_point() -> None:
    assert (
        operating_point(_r("baseline_hf", "none", "cold", tput=None), _rc(), 32, _RESIDENT) is None
    )
