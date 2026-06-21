"""Per-token energy from a measured ``RunResult`` (the on-prem OPEX driver, D5/D6).

GPU energy is measured (NVML integration); CPU energy is estimated from TDP
because the harness records only GPU joules (``cpu_energy_j_est`` is null on the
committed runs). Both are reduced to *per decoded token* via the steady decode
rate (``tpot_s``), which excludes the one-off load/prefill cost from the unit
energy — the right basis for steady-state cost per token.
"""

from __future__ import annotations

from airllm_bench.harness.result import RunResult


def avg_power_w(energy_j: float, runtime_s: float) -> float:
    """Average power over a run = integrated energy / wall-clock."""
    return energy_j / runtime_s


def gpu_energy_per_token_j(result: RunResult) -> float | None:
    """Measured GPU joules per decoded token (avg power x per-token decode time).

    None when the run lacks the energy/timing to derive it (e.g. a clean OOM).
    """
    if not result.tpot_s or not result.gpu_energy_j or not result.runtime_s:
        return None
    return avg_power_w(result.gpu_energy_j, result.runtime_s) * result.tpot_s


def cpu_energy_per_token_j(tpot_s: float, cpu_tdp_w: float, cpu_load_fraction: float) -> float:
    """Estimated CPU joules per decoded token = TDP x load-fraction x decode time."""
    return cpu_tdp_w * cpu_load_fraction * tpot_s


def energy_per_token_j(
    result: RunResult,
    cpu_tdp_w: float,
    cpu_load_fraction: float,
) -> float | None:
    """Total joules per decoded token = measured GPU + estimated CPU.

    None when the GPU term cannot be derived (no measured energy/timing).
    """
    gpu = gpu_energy_per_token_j(result)
    if gpu is None or result.tpot_s is None:
        return None
    return gpu + cpu_energy_per_token_j(result.tpot_s, cpu_tdp_w, cpu_load_fraction)
