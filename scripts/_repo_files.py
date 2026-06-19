"""Enumerate repo files via git, so `.gitignore` is the single source of truth.

`git ls-files` lists only tracked + staged files, so anything ignored — the
Claude Code worktrees under `.claude/`, `.venv`, build caches, future dirs —
is excluded automatically. No hand-maintained denylist to fall behind reality
(which is exactly how the worktree double-scan bug crept in). This mirrors how
`ruff` already behaves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_ls_files(*patterns: str, root: Path) -> list[Path]:
    """Absolute paths of tracked/staged files under *root* matching *patterns*.

    Patterns are git pathspecs (e.g. ``"*.py"``), matched at any depth. Raises
    ``subprocess.CalledProcessError`` if *root* is not inside a git repo.
    """
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--", *patterns],
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / rel for rel in result.stdout.split("\0") if rel]
