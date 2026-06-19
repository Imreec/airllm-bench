"""Forbid the config-owned model repo id from being hardcoded in ``src/``.

HW5's single source of truth for the model under test is ``config/setup.json``;
``src/`` code must load it via ``SetupConfig``, never paste the repo id as a
string literal. Mirrors HW2's "defined once" check, repointed at HW5's registry.
No-op until a ``src/`` module references the model (empty-skeleton friendly).

File enumeration goes through git (``_repo_files``) so ignored sibling
checkouts — Claude Code worktrees — are never scanned. Only ``src/`` is checked,
so throwaway ``spike/`` scripts may use the literal freely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _repo_files import git_ls_files

ROOT = Path(__file__).parent.parent
SETUP = ROOT / "config" / "setup.json"


def watched_repo_id() -> str | None:
    """The model repo id that must not appear hardcoded in ``src/`` (or None)."""
    if not SETUP.exists():
        return None
    data = json.loads(SETUP.read_text(encoding="utf-8"))
    repo_id = data.get("model", {}).get("repo_id")
    return repo_id if isinstance(repo_id, str) and repo_id else None


def main() -> int:
    repo_id = watched_repo_id()
    if not repo_id:
        return 0
    offenders: list[Path] = []
    for path in git_ls_files("*.py", root=ROOT):
        if "src/" not in str(path).replace("\\", "/"):
            continue
        try:
            if repo_id in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(ROOT))
        except (UnicodeDecodeError, PermissionError):
            continue
    if offenders:
        print(
            f"FAIL: model repo id {repo_id!r} hardcoded in src/ — load it from config/setup.json:"
        )
        for o in offenders:
            print(f"  {o}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
