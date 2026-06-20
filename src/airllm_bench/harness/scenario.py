"""Run exactly ONE scenario from config and persist its RunResult (T5.4, PLAN §4).

The matrix orchestrator (T5.4b) spawns this per scenario in a fresh subprocess, so a
true *cold* run gets an unwarmed page cache that an in-process loop could never give.
Keyless: with the mock runner and an injected sampler it exercises the whole
build → run → write path with no GPU. The sampler/env (hardware-bound) are injected by
the orchestrator, keeping this module itself importable and testable without NVML/torch.
"""

from __future__ import annotations

import json
from pathlib import Path

from airllm_bench.harness.result import RunResult
from airllm_bench.harness.run import Sampler, run
from airllm_bench.runners.factory import build_runner
from airllm_bench.shared.config_models import ExperimentConfig, SetupConfig


def _prompt_tokens(exp: ExperimentConfig) -> int:
    """Native token count if the matrix pinned it, else a keyless whitespace estimate."""
    if exp.prompt_tokens is not None:
        return exp.prompt_tokens
    return len(exp.prompt.split())


def append_result(result: RunResult, results_dir: Path) -> Path:
    """Append one RunResult as a JSON line to ``results/<exp_id>.jsonl`` (the source of truth)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result.exp_id}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.model_dump()) + "\n")
    return path


def run_scenario(
    exp: ExperimentConfig,
    setup: SetupConfig,
    *,
    exp_id: str,
    sampler: Sampler | None = None,
    env: dict[str, str] | None = None,
    results_dir: Path | None = None,
) -> RunResult:
    """Build the runner, run the single scenario, and (optionally) persist the result."""
    runner = build_runner(exp.runner, setup)
    result = run(
        runner,
        exp,
        exp_id=exp_id,
        model=setup.model.repo_id,
        param_count=setup.model.params,
        prompt_tokens=_prompt_tokens(exp),
        sampler=sampler,
        env=env or {},
    )
    if results_dir is not None:
        append_result(result, results_dir)
    return result
