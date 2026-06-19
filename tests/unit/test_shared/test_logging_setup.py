"""Tests for the rotating JSONL run logger."""

from __future__ import annotations

import json
from pathlib import Path

from airllm_bench.shared.logging_setup import build_logger, log_event


def test_writes_jsonl_and_redacts_secret(tmp_path: Path) -> None:
    lf = tmp_path / "run.jsonl"
    logger = build_logger(lf, name="t_redact")
    log_event(logger, {"metric": "ttft", "value": 1.2, "api_key": "sk-secret"})
    rec = json.loads(lf.read_text(encoding="utf-8").strip())
    assert rec["event"]["value"] == 1.2
    assert rec["event"]["api_key"] == "***redacted***"
    assert "timestamp" in rec
    assert rec["level"] == "INFO"


def test_nested_redaction(tmp_path: Path) -> None:
    lf = tmp_path / "run2.jsonl"
    logger = build_logger(lf, name="t_nested")
    log_event(logger, {"outer": {"password": "p", "ok": 1}})
    rec = json.loads(lf.read_text(encoding="utf-8").strip())
    assert rec["event"]["outer"]["password"] == "***redacted***"
    assert rec["event"]["outer"]["ok"] == 1


def test_non_mapping_message(tmp_path: Path) -> None:
    lf = tmp_path / "msg.jsonl"
    logger = build_logger(lf, name="t_msg")
    logger.info("plain string")
    rec = json.loads(lf.read_text(encoding="utf-8").strip())
    assert rec["event"]["message"] == "plain string"


def test_line_count_rotation(tmp_path: Path) -> None:
    lf = tmp_path / "roll.jsonl"
    logger = build_logger(lf, name="t_roll", max_files=2, max_lines=2)
    for i in range(5):
        log_event(logger, {"i": i})
    current = lf.read_text(encoding="utf-8").strip().splitlines()
    assert len(current) <= 2
    assert (tmp_path / "roll.jsonl.1").exists()


def test_no_rotation_when_max_lines_zero(tmp_path: Path) -> None:
    lf = tmp_path / "flat.jsonl"
    logger = build_logger(lf, name="t_zero", max_lines=0)
    for i in range(4):
        log_event(logger, {"i": i})
    assert len(lf.read_text(encoding="utf-8").strip().splitlines()) == 4
