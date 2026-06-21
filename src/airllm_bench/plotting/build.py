"""Build every report figure from committed data (the ``make figures`` entry).

Loads results + the two config files, prepares each figure's inputs (``prepare``),
renders via the builders, and writes PNGs under ``figures/``. Pure of any model or
network — regenerates the committed figures offline (D11).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from airllm_bench.economics.breakeven import build_lines
from airllm_bench.economics.config import EconomicsConfig
from airllm_bench.harness.result import RunResult
from airllm_bench.metrics.load import load_results
from airllm_bench.plotting import prepare
from airllm_bench.plotting.base import save_figure
from airllm_bench.plotting.economics_fig import breakeven_fig
from airllm_bench.plotting.latency import cold_warm_ttft_fig, itl_fig, ttft_vs_length_fig
from airllm_bench.plotting.roofline_fig import roofline_fig
from airllm_bench.plotting.throughput_quality import perplexity_fig, throughput_fig
from airllm_bench.roofline.points import (
    OperatingPoint,
    operating_point,
    prefill_operating_point,
    resident_runner_names,
)
from airllm_bench.shared.config_models import RooflineConfig, SetupConfig

_VOLUMES = [1e6, 1e7, 1e8, 1e9, 1e10]


def _operating_points(
    results: Sequence[RunResult],
    rc: RooflineConfig,
    setup: SetupConfig,
) -> list[OperatingPoint]:
    """Decode points for every run + prefill points where a TTFT exists."""
    resident = resident_runner_names(setup.runners)
    ram = setup.hardware.ram_gb
    pts: list[OperatingPoint] = []
    for r in results:
        decode = operating_point(r, rc, ram, resident)
        prefill = prefill_operating_point(r, rc, ram, resident)
        pts.extend(p for p in (decode, prefill) if p is not None)
    return pts


def build_figures(
    results_dir: str | Path,
    setup_path: str | Path,
    econ_path: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Render all figures from committed data; return the written paths."""
    results = load_results(results_dir)
    setup = SetupConfig.from_file(setup_path)
    econ = EconomicsConfig.from_file(econ_path)
    rc = setup.roofline
    if rc is None:
        msg = "setup.json has no roofline block — cannot build the roofline figure"
        raise ValueError(msg)
    out = Path(out_dir)
    paths = [
        save_figure(ttft_vs_length_fig(prepare.ttft_curve(results)), out / "ttft_vs_length.png"),
        save_figure(itl_fig(prepare.itl_series(results)), out / "itl.png"),
        save_figure(
            cold_warm_ttft_fig(prepare.cold_warm_ttft(results)), out / "cold_warm_ttft.png"
        ),
        save_figure(
            throughput_fig(prepare.throughput_by_scenario(results)), out / "throughput.png"
        ),
        save_figure(perplexity_fig(prepare.perplexity_by_quant(results)), out / "perplexity.png"),
        save_figure(roofline_fig(rc, _operating_points(results, rc, setup)), out / "roofline.png"),
    ]
    realistic = prepare.realistic_run(results)
    airllm = prepare.cautionary_airllm_run(results)
    if realistic and airllm and realistic.max_new_tokens:
        ratio = realistic.prompt_tokens / realistic.max_new_tokens
        lines = build_lines(econ, econ.active_model, realistic, airllm, ratio)
        paths.append(save_figure(breakeven_fig(lines, _VOLUMES), out / "breakeven.png"))
    return paths
