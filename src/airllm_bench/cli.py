"""Thin CLI: drive the benchmark matrix (T5.4). Figures/economics land in Phase 6.

``airllm-bench bench --experiments a.json b.json ...`` runs the matrix orchestrator over
the given single-scenario configs. The heavy wiring (subprocess spawn, cold-cache flush)
is built here from config and passed into the keyless-tested orchestrator (PLAN §2: no
SDK facade — the CLI calls plain module functions).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from airllm_bench.harness.cache_flush import build_cold_flush
from airllm_bench.harness.orchestrator import run_matrix
from airllm_bench.shared.config_models import SetupConfig

_SCENARIO_MODULE = "airllm_bench.harness.scenario_main"


def _spawn_subprocess(exp_path: Path, setup_path: Path, exp_id: str, results_dir: Path) -> int:
    """Run one scenario in a fresh process (true cold cache + bounded peak-memory accounting)."""
    cmd = [
        sys.executable,
        "-m",
        _SCENARIO_MODULE,
        str(exp_path),
        str(setup_path),
        exp_id,
        str(results_dir),
    ]
    return subprocess.run(cmd, check=False).returncode


def _bench(args: argparse.Namespace, *, orchestrate: Callable[..., list[int]] = run_matrix) -> int:
    """Build the spawn + cold-flush wiring from config and run the matrix."""
    setup_path = Path(args.setup)
    setup = SetupConfig.from_file(setup_path)
    cold_flush = None if args.no_flush else build_cold_flush(setup.cold_flush)
    codes = orchestrate(
        [Path(p) for p in args.experiments],
        setup_path,
        results_dir=Path(args.results_dir),
        spawn=_spawn_subprocess,
        cold_flush=cold_flush,
    )
    return 1 if any(codes) else 0


def build_parser() -> argparse.ArgumentParser:
    """The ``airllm-bench`` argument parser (only ``bench`` for now)."""
    parser = argparse.ArgumentParser(prog="airllm-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("bench", help="run the scenario matrix")
    bench.add_argument("--experiments", nargs="+", required=True)
    bench.add_argument("--setup", default="config/setup.json")
    bench.add_argument("--results-dir", default="results")
    bench.add_argument("--no-flush", action="store_true", help="skip cold-cache flush (warm-only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args and dispatch."""
    args = build_parser().parse_args(argv)
    return _bench(args)


if __name__ == "__main__":  # pragma: no cover — process entrypoint
    raise SystemExit(main())
