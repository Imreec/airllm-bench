"""CI gate: assert README.md satisfies the report contract (T7.1, D11).

Thin wrapper over ``airllm_bench.report.contract.audit`` — reads the committed
README and fails loud if any mandated section heading or committed figure is
missing. Wired into ``make grade`` once the report lands (T7.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

from airllm_bench.report.contract import audit

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def main() -> int:
    """Audit README.md; print every gap and return non-zero if the report is incomplete."""
    if not README.exists():
        print(f"FAIL: {README} does not exist")
        return 1
    problems = audit(README.read_text(encoding="utf-8"))
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        print(f"\n{len(problems)} report-contract problem(s).")
        return 1
    print("check_report: README satisfies the report contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
