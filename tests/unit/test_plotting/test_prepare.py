"""Tests for the plotting data-prep selectors (pure, from RunResults)."""

from __future__ import annotations

from airllm_bench.harness.result import RunResult
from airllm_bench.plotting.prepare import (
    cautionary_airllm_run,
    cold_warm_ttft,
    itl_series,
    perplexity_by_quant,
    realistic_run,
    throughput_by_scenario,
    ttft_curve,
)


def _r(exp: str, runner: str, quant: str, phase: str, **over: object) -> RunResult:
    base: dict[str, object] = {
        "exp_id": exp,
        "runner": runner,
        "quant": quant,
        "model": "m",
        "param_count": 1,
        "prompt_tokens": 40,
        "max_new_tokens": 10,
        "phase": phase,
        "ok": True,
    }
    base.update(over)
    return RunResult.model_validate(base)


def _sweep() -> list[RunResult]:
    return [
        _r("airllm-nf4-cold", "airllm", "nf4", "cold", ttft_s=43.7, throughput_tok_s=0.04),
        _r(
            "airllm-nf4-warm",
            "airllm",
            "nf4",
            "warm",
            ttft_s=19.3,
            throughput_tok_s=0.05,
            perplexity=39.9,
        ),
        _r("airllm-nf4-len64", "airllm", "nf4", "warm", prompt_tokens=64, ttft_s=20.0),
        _r("airllm-nf4-len256", "airllm", "nf4", "warm", prompt_tokens=256, ttft_s=21.0),
        _r("airllm-int8-cold", "airllm", "int8", "cold", ttft_s=54.3),
        _r(
            "airllm-int8-warm",
            "airllm",
            "int8",
            "warm",
            ttft_s=50.8,
            throughput_tok_s=0.02,
            perplexity=24.2,
        ),
    ]


def test_ttft_curve_maps_prompt_length_to_ttft_for_nf4_warm() -> None:
    assert ttft_curve(_sweep()) == {40: 19.3, 64: 20.0, 256: 21.0}


def test_cold_warm_ttft_pairs_each_quant() -> None:
    cw = cold_warm_ttft(_sweep())
    assert cw["4-bit"] == (43.7, 19.3)
    assert cw["8-bit"] == (54.3, 50.8)


def test_cold_warm_pairs_at_a_symmetric_baseline_length() -> None:
    # A stray long-prompt cold run must NOT be paired against the 40-token warm run:
    # the baseline (min length) is applied to BOTH phases.
    rows = [
        _r("nf4-cold-256", "airllm", "nf4", "cold", prompt_tokens=256, ttft_s=99.0),
        _r("nf4-cold-40", "airllm", "nf4", "cold", prompt_tokens=40, ttft_s=43.7),
        _r("nf4-warm-40", "airllm", "nf4", "warm", prompt_tokens=40, ttft_s=19.3),
    ]
    assert cold_warm_ttft(rows)["4-bit"] == (43.7, 19.3)  # the 256-token cold is ignored


def test_throughput_skips_runs_without_a_rate() -> None:
    tput = throughput_by_scenario(_sweep())
    assert "airllm-nf4-warm" in tput
    assert "airllm-nf4-len64" not in tput  # no throughput recorded


def test_perplexity_by_quant_uses_labelled_quants() -> None:
    ppl = perplexity_by_quant(_sweep())
    assert ppl == {"4-bit": 39.9, "8-bit": 24.2}


def test_perplexity_compares_quants_at_the_same_baseline_length() -> None:
    # A longer-prompt sweep run carries a different (lower) perplexity, but perplexity
    # across prompt lengths is not comparable. The cross-quant figure must pin to the
    # baseline (shortest) length — like every sibling selector — not pick whichever
    # warm run loads first (which would plot 1.7 for 4-bit and invert the thesis).
    rows = [
        _r("nf4-len256", "airllm", "nf4", "warm", prompt_tokens=256, perplexity=1.7),
        _r("nf4-warm", "airllm", "nf4", "warm", prompt_tokens=40, perplexity=39.9),
    ]
    assert perplexity_by_quant(rows) == {"4-bit": 39.9}


def test_itl_series_picks_nf4_cold_and_warm() -> None:
    rows = [
        _r("airllm-nf4-cold", "airllm", "nf4", "cold", itl_s=[3.0, 1.0]),
        _r("airllm-nf4-warm", "airllm", "nf4", "warm", itl_s=[1.0, 1.0]),
    ]
    assert itl_series(rows) == {"nf4 cold": [3.0, 1.0], "nf4 warm": [1.0, 1.0]}


def test_run_selectors_find_the_economics_lines() -> None:
    rows = [
        _r("llamacpp-q4-warm", "llamacpp", "q4_k_m", "warm"),
        _r("airllm-nf4-warm", "airllm", "nf4", "warm"),
    ]
    assert realistic_run(rows).exp_id == "llamacpp-q4-warm"  # type: ignore[union-attr]
    assert cautionary_airllm_run(rows).exp_id == "airllm-nf4-warm"  # type: ignore[union-attr]
    assert realistic_run([]) is None
