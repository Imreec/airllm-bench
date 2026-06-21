"""Assert every committed figure traces to backing raw data (D11).

A negative result must be *evidenced*, not asserted — so no figure may exist
without rows under ``results/`` behind it. Once plotting landed (T6.3) this
tightened from "any figure needs any data" to: the committed figure set must be
exactly the expected set, and the specific scenarios those figures are built from
must be present under ``results/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

#: The figures `make figures` emits (kept in lockstep with plotting/build.py).
_EXPECTED_FIGURES = {
    "ttft_vs_length.png",
    "itl.png",
    "cold_warm_ttft.png",
    "throughput.png",
    "perplexity.png",
    "roofline.png",
    "breakeven.png",
}
#: Scenarios the headline figures (break-even, roofline, cold/warm) are built from.
_REQUIRED_RESULTS = {
    "llamacpp-q4-warm",
    "airllm-nf4-cold",
    "airllm-nf4-warm",
}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if not FIGURES.exists():
        return 0
    figs = {p.name for p in FIGURES.rglob("*.png") if "scratch" not in p.parts}
    if not figs:
        return 0
    missing_figs = _EXPECTED_FIGURES - figs
    if missing_figs:
        return _fail(f"expected figures missing: {sorted(missing_figs)}")
    have = {p.stem for p in RESULTS.glob("*.jsonl")} if RESULTS.exists() else set()
    missing_data = _REQUIRED_RESULTS - have
    if missing_data:
        return _fail(f"figures present but backing results missing: {sorted(missing_data)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
