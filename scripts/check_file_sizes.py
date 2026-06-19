"""Check that no Python source file exceeds 150 lines.

Per CLAUDE.md §1, every Python file must be <= 150 source lines.
This includes test files. Run as a pre-commit hook and in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES = 150
SCAN_DIRS = ("src", "tests", "scripts")
EXCLUDE_PATTERNS = ("__pycache__", ".pyc")


def count_source_lines(path: Path) -> int:
    """Count non-blank, non-comment-only lines."""
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            total += 1
    return total


def find_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for scan_dir in SCAN_DIRS:
        full = root / scan_dir
        if not full.exists():
            continue
        for path in full.rglob("*.py"):
            if any(p in str(path) for p in EXCLUDE_PATTERNS):
                continue
            files.append(path)
    return files


def main() -> int:
    root = Path(__file__).parent.parent
    violations: list[tuple[Path, int]] = []
    for path in find_python_files(root):
        n = count_source_lines(path)
        if n > MAX_LINES:
            violations.append((path.relative_to(root), n))
    if violations:
        print(f"FAIL: {len(violations)} file(s) exceed {MAX_LINES} source lines:")
        for path, n in sorted(violations, key=lambda x: -x[1]):
            print(f"  {n:>4} lines  {path}")
        print()
        print("Per CLAUDE.md, refactor any file >150 lines into smaller pieces.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
