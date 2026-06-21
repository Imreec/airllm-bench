"""Thin CLI: drive the benchmark matrix (``bench``) and render figures (``figures``).

``airllm-bench bench --experiments a.json ...`` runs the matrix orchestrator;
``airllm-bench figures`` regenerates every report figure from committed data
(keyless, offline). The CLI builds wiring from config and calls plain module
functions (PLAN §2: no SDK facade).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from airllm_bench.harness.cache_flush import build_cold_flush
from airllm_bench.harness.orchestrator import run_matrix
from airllm_bench.plotting.build import build_figures
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


def _figures(args: argparse.Namespace) -> int:
    """Regenerate every report figure from committed data (keyless, offline)."""
    paths = build_figures(args.results_dir, args.setup, args.economics, args.out)
    for p in paths:
        print(p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``airllm-bench`` argument parser (``bench`` + ``figures``)."""
    parser = argparse.ArgumentParser(prog="airllm-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("bench", help="run the scenario matrix")
    bench.add_argument("--experiments", nargs="+", required=True)
    bench.add_argument("--setup", default="config/setup.json")
    bench.add_argument("--results-dir", default="results")
    bench.add_argument("--no-flush", action="store_true", help="skip cold-cache flush (warm-only)")
    figs = sub.add_parser("figures", help="regenerate report figures from results/")
    figs.add_argument("--results-dir", default="results")
    figs.add_argument("--setup", default="config/setup.json")
    figs.add_argument("--economics", default="config/economics.json")
    figs.add_argument("--out", default="figures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args and dispatch."""
    args = build_parser().parse_args(argv)
    return _figures(args) if args.cmd == "figures" else _bench(args)


if __name__ == "__main__":  # pragma: no cover — process entrypoint
    raise SystemExit(main())
