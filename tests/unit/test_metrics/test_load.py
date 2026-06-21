"""Tests for the results loader (reads committed JSONL into RunResults)."""

from __future__ import annotations

from pathlib import Path

import pytest

from airllm_bench.harness.result import RunResult
from airllm_bench.metrics.load import index_by_exp_id, load_results


def _write(path: Path, exp_id: str, ok: bool = True) -> None:
    r = RunResult(
        exp_id=exp_id,
        runner="airllm",
        quant="nf4",
        model="m",
        param_count=1,
        prompt_tokens=40,
        max_new_tokens=10,
        phase="cold",
        ok=ok,
    )
    path.write_text(r.model_dump_json() + "\n", encoding="utf-8")


def test_load_results_reads_every_jsonl_file(tmp_path: Path) -> None:
    _write(tmp_path / "a.jsonl", "a")
    _write(tmp_path / "b.jsonl", "b")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    results = load_results(tmp_path)

    assert {r.exp_id for r in results} == {"a", "b"}
    assert all(isinstance(r, RunResult) for r in results)


def test_load_results_sorted_by_exp_id_for_determinism(tmp_path: Path) -> None:
    _write(tmp_path / "z.jsonl", "z")
    _write(tmp_path / "a.jsonl", "a")
    assert [r.exp_id for r in load_results(tmp_path)] == ["a", "z"]


def test_load_results_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "nope")


def test_index_by_exp_id_maps_id_to_result(tmp_path: Path) -> None:
    _write(tmp_path / "a.jsonl", "a")
    idx = index_by_exp_id(load_results(tmp_path))
    assert idx["a"].exp_id == "a"
