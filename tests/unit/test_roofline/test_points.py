"""Tests for operating-point placement on the hierarchical roofline (D7)."""

from __future__ import annotations

import pytest

from airllm_bench.harness.result import RunResult
from airllm_bench.roofline.points import fits_in_ram, operating_point
from airllm_bench.shared.config_models import RooflineConfig

_P = 32_000_000_000


def _rc() -> RooflineConfig:
    return RooflineConfig(
        compute_fp16_tflops=68.2,
        hbm_gb_s=912,
        pcie_gb_s=31.5,
        nvme_gb_s=7.0,
        bytes_per_weight={"none": 2.0, "int8": 1.0, "nf4": 0.5, "q4_k_m": 0.56},
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
        }
    )


def test_fits_in_ram_is_footprint_under_capacity() -> None:
    assert fits_in_ram(_P, 0.5, ram_gb=32) is True  # nf4 = 16 GB
    assert fits_in_ram(_P, 2.0, ram_gb=32) is False  # fp16 = 64 GB


def test_operating_point_intensity_and_per_token_arithmetic() -> None:
    pt = operating_point(_r("airllm", "nf4", "warm"), _rc(), ram_gb=32)
    assert pt is not None
    assert pt.flops_per_token == pytest.approx(2 * _P)  # 64e9
    assert pt.bytes_per_token == pytest.approx(_P * 0.5)  # 16e9
    assert pt.intensity_flops_per_byte == pytest.approx(4.0)  # 2 / 0.5
    assert pt.achieved_flops == pytest.approx(2 * _P * 0.05)


def test_resident_runtimes_bind_on_hbm() -> None:
    assert operating_point(_r("llamacpp", "q4_k_m", "warm"), _rc(), 32).binding_tier == "hbm"
    assert operating_point(_r("baseline_hf", "none", "cold"), _rc(), 32).binding_tier == "hbm"


def test_airllm_cold_binds_on_nvme() -> None:
    assert operating_point(_r("airllm", "nf4", "cold"), _rc(), 32).binding_tier == "nvme"


def test_airllm_warm_climbs_to_pcie_only_when_it_fits_ram() -> None:
    # nf4 (16 GB) fits the 32 GB cache -> PCIe/RAM; fp16 (64 GB) does not -> still NVMe.
    assert operating_point(_r("airllm", "nf4", "warm"), _rc(), 32).binding_tier == "pcie"
    assert operating_point(_r("airllm", "none", "warm"), _rc(), 32).binding_tier == "nvme"


def test_failed_run_has_no_operating_point() -> None:
    assert operating_point(_r("baseline_hf", "none", "cold", tput=None), _rc(), 32) is None
