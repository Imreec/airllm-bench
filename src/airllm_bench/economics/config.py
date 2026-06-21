"""Typed model for ``config/economics.json`` (D6/D8 — documented constants only).

Lives here, where it is consumed, per PLAN §2. Every cost number flows from this
config: no prices, tariffs, or rates are hardcoded in ``src/``. Sections allow
extra keys (``source``/``url``/``retrieved``/``_note`` provenance) so the audit
trail rides along without a schema change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from airllm_bench.economics.costing import TokenRate
from airllm_bench.shared.config import load_versioned


class _Section(BaseModel):
    """Base for config sections: keep unknown (provenance) keys."""

    model_config = ConfigDict(extra="allow")


class ApiRate(_Section):
    """Per-model API price, input and output billed separately (USD per Mtok)."""

    input_per_mtok_usd: float
    output_per_mtok_usd: float


class Electricity(_Section):
    tariff_usd_per_kwh: float


class HardwarePower(_Section):
    """Power draw knobs. ``cpu_load_fraction`` scales TDP to the share actually
    drawn during inference (results carry only measured GPU energy)."""

    gpu_tdp_w: float
    cpu_tdp_w: float
    cpu_load_fraction: float = 0.5


class Capex(_Section):
    """Up-front hardware cost. ``scope`` selects which figure feeds the on-prem
    line (D6 exposed knob): the GPU alone or the whole build."""

    scope: str
    gpu_cost_usd: float
    whole_pc_cost_usd: float
    amortization_years: float


class CloudGpu(_Section):
    usd_per_hour: float


class Caching(_Section):
    """``cached_input_discount`` = fraction of the input rate billed on a hit;
    ``cached_fraction`` = share of input tokens served from cache."""

    cached_input_discount: float
    cached_fraction: float = 0.5


class Utilization(_Section):
    """``duty_cycle`` = fraction of wall-clock the owned box serves inference (the
    most sensitive on-prem knob, D6)."""

    duty_cycle: float


class EconomicsConfig(BaseModel):
    """``config/economics.json`` — all cost constants with provenance."""

    model_config = ConfigDict(extra="ignore")

    version: str
    api: dict[str, ApiRate]
    electricity: Electricity
    hardware_power: HardwarePower
    capex: Capex
    cloud_gpu: CloudGpu
    caching: Caching
    utilization: Utilization
    raw_data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> EconomicsConfig:
        """Load and validate ``config/economics.json`` (fail-loud on version)."""
        return load_versioned(path, cls)

    def token_rate(self, model_key: str) -> TokenRate:
        """The API price for ``model_key`` as a ``TokenRate`` (per-Mtok)."""
        r = self.api[model_key]
        return TokenRate(input_per_mtok=r.input_per_mtok_usd, output_per_mtok=r.output_per_mtok_usd)

    def capex_usd(self) -> float:
        """The CAPEX figure selected by ``capex.scope`` (gpu_only | whole_pc)."""
        return (
            self.capex.gpu_cost_usd
            if self.capex.scope == "gpu_only"
            else self.capex.whole_pc_cost_usd
        )
