"""Load committed ``results/*.jsonl`` into ``RunResult`` objects.

The single read path for every Tier-1 analysis module (metrics/economics/
roofline/plotting): they consume committed JSON and never touch a model. Each
file holds one scenario as one JSON line (written by ``harness.scenario``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from airllm_bench.harness.result import RunResult


def load_results(results_dir: str | Path) -> list[RunResult]:
    """Read every ``*.jsonl`` under ``results_dir`` into ``RunResult``s.

    Sorted by ``exp_id`` so downstream figures are deterministic regardless of
    filesystem ordering.

    Raises:
        FileNotFoundError: when ``results_dir`` does not exist.
    """
    root = Path(results_dir)
    if not root.is_dir():
        msg = f"results dir not found: {root}"
        raise FileNotFoundError(msg)
    results = [
        RunResult.model_validate_json(line)
        for path in root.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(results, key=lambda r: r.exp_id)


def index_by_exp_id(results: Sequence[RunResult]) -> dict[str, RunResult]:
    """Map ``exp_id`` → ``RunResult`` for keyed lookup by the analysis lines."""
    return {r.exp_id: r for r in results}
