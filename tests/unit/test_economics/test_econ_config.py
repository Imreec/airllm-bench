"""Tests for the economics config model (typed, versioned, scope-aware)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from airllm_bench.economics.config import EconomicsConfig
from airllm_bench.economics.costing import TokenRate
from airllm_bench.shared.config import ConfigVersionError

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_committed_economics_json_loads_and_is_versioned() -> None:
    cfg = EconomicsConfig.from_file(REPO_ROOT / "config" / "economics.json")
    assert cfg.version.startswith("1.")
    assert cfg.electricity.tariff_usd_per_kwh > 0


def test_token_rate_exposes_the_api_anchor(economics_config_data: dict[str, Any]) -> None:
    cfg = EconomicsConfig.model_validate({**economics_config_data, "raw_data": {}})
    rate = cfg.token_rate("qwen2.5-32b")
    assert rate == TokenRate(input_per_mtok=0.5, output_per_mtok=1.0)


def test_active_model_selects_the_priced_line(economics_config_data: dict[str, Any]) -> None:
    cfg = EconomicsConfig.model_validate({**economics_config_data, "raw_data": {}})
    assert cfg.active_model == "qwen2.5-32b"
    assert cfg.token_rate(cfg.active_model) == TokenRate(input_per_mtok=0.5, output_per_mtok=1.0)


def test_committed_economics_json_active_model_is_priced() -> None:
    cfg = EconomicsConfig.from_file(REPO_ROOT / "config" / "economics.json")
    assert cfg.active_model in cfg.api  # the active key must have a rate


def test_capex_usd_selects_by_scope(economics_config_data: dict[str, Any]) -> None:
    gpu = EconomicsConfig.model_validate({**economics_config_data, "raw_data": {}})
    assert gpu.capex_usd() == 1200
    whole = {
        **economics_config_data,
        "capex": {**economics_config_data["capex"], "scope": "whole_pc"},
    }
    assert EconomicsConfig.model_validate({**whole, "raw_data": {}}).capex_usd() == 2400


def test_version_mismatch_fails_loud(
    economics_config_data: dict[str, Any],
    write_config: Any,
) -> None:
    bad = write_config({**economics_config_data, "version": "2.00"}, "economics.json")
    with pytest.raises(ConfigVersionError):
        EconomicsConfig.from_file(bad)
