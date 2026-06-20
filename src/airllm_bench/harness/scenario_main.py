"""Subprocess entrypoint: run ONE scenario on real hardware and persist it (T5.4).

The orchestrator spawns ``python -m airllm_bench.harness.scenario_main <exp> <setup>
<exp_id> <results_dir>`` per scenario, so each cold run gets a fresh process + cache.
This is where the hardware instrumentation (NVML sampler + env metadata) is wired into
the otherwise-keyless scenario worker. Keyless-testable by injecting fake pynvml/psutil
and a mock-runner experiment; only the ``__main__`` shim is excluded from coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

from airllm_bench.harness.env_info import collect_env
from airllm_bench.harness.hw_reader import make_reader
from airllm_bench.harness.sampler import ResourceSampler
from airllm_bench.harness.scenario import run_scenario
from airllm_bench.shared.config_models import ExperimentConfig, SetupConfig


def main(argv: list[str]) -> int:
    """Load the configs, run the one scenario under a live sampler, write its RunResult."""
    exp_path, setup_path, exp_id, results_dir = argv
    exp = ExperimentConfig.from_file(exp_path)
    setup = SetupConfig.from_file(setup_path)
    sampler = ResourceSampler(make_reader())
    run_scenario(
        exp,
        setup,
        exp_id=exp_id,
        sampler=sampler,
        env=collect_env(),
        results_dir=Path(results_dir),
    )
    # A clean baseline OOM is a recorded result (ok=False), not a process failure.
    return 0


if __name__ == "__main__":  # pragma: no cover — process entrypoint, exercised on the box
    sys.exit(main(sys.argv[1:]))
