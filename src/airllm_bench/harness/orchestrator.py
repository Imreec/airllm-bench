"""Drive the scenario matrix with subprocess isolation + cold-cache flush (T5.4, PLAN §4).

Each scenario runs in a *fresh subprocess* so a true cold run gets an unwarmed page
cache an in-process loop could never provide; before every ``cold`` scenario we flush
the OS standby list. Pairing is encoded by ORDER in the matrix (a cold scenario
immediately followed by its warm twin on the now-populated cache). The flush and the
subprocess spawn are injected, so the whole ordering/flush logic is keyless-tested.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from tqdm import tqdm

from airllm_bench.shared.config_models import ExperimentConfig, SetupConfig

Spawn = Callable[[Path, Path, str, Path], int]
Progress = Callable[[Iterable[Path]], Iterable[Path]]


def run_matrix(
    scenario_paths: Sequence[Path],
    setup_path: Path,
    *,
    results_dir: Path,
    spawn: Spawn,
    cold_flush: Callable[[], None] | None = None,
    progress: Progress = tqdm,
) -> list[int]:
    """Run every scenario config in order; flush before each cold one. Returns exit codes."""
    SetupConfig.from_file(setup_path)  # validate once up front — fail before any subprocess
    codes: list[int] = []
    for path in progress(list(scenario_paths)):
        exp = ExperimentConfig.from_file(path)
        if exp.phase == "cold" and cold_flush is not None:
            cold_flush()
        codes.append(spawn(path, setup_path, path.stem, results_dir))
    return codes
