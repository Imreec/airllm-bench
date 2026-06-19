"""Assert every committed figure traces to backing raw data (D11).

A negative result must be *evidenced*, not asserted — so no figure may exist
without rows under ``results/`` behind it. No-op until figures land (Phase 6+),
mirroring an empty skeleton; the per-figure mapping is tightened when plotting
exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main() -> int:
    if not FIGURES.exists():
        return 0
    figs = [p for p in FIGURES.rglob("*.png") if "scratch" not in p.parts]
    if not figs:
        return 0
    has_data = RESULTS.exists() and any(RESULTS.rglob("*.json")) or any(RESULTS.rglob("*.jsonl"))
    if not has_data:
        print(f"FAIL: {len(figs)} figure(s) present but no results/ raw data backs them.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
