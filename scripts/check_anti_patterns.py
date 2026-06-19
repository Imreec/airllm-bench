"""Detect anti-patterns that we never want to slip into committed code.

Catches the specific failures from the previous repo audit:
- NotImplementedError stubs in concrete code
- /mnt/user-data/ paths leaked into code/config (allowed in markdown which discusses them)
- "AI Agent" listed as author in pyproject.toml
- Python version mismatch between pyproject and ruff/mypy
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from _repo_files import git_ls_files

ROOT = Path(__file__).parent.parent

# Patterns forbidden in code/config (not markdown — markdown legitimately discusses them)
FORBIDDEN_IN_CODE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/mnt/user-data"), "Leaked Claude internal path '/mnt/user-data'"),
    (re.compile(r"/mnt/skills"), "Leaked Claude internal path '/mnt/skills'"),
)
CODE_EXTS = (".py", ".toml", ".yaml", ".yml")
# This file holds the forbidden patterns as literals, so it must not scan itself.
EXCLUDE_FILES = ("check_anti_patterns.py",)


def find_code_files(root: Path) -> list[Path]:
    """Tracked code/config files (git-enumerated, so ignored dirs are skipped)."""
    patterns = tuple(f"*{ext}" for ext in CODE_EXTS)
    return [p for p in git_ls_files(*patterns, root=root) if p.name not in EXCLUDE_FILES]


def check_forbidden(files: list[Path]) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            for pattern, description in FORBIDDEN_IN_CODE:
                if pattern.search(line):
                    violations.append((path.relative_to(ROOT), line_no, description))
    return violations


def check_pyproject_authors() -> list[tuple[Path, int, str]]:
    """Verify pyproject authors don't contain placeholders."""
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return []
    try:
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError:
        return []
    authors = data.get("project", {}).get("authors", [])
    violations: list[tuple[Path, int, str]] = []
    for author in authors:
        name = author.get("name", "") if isinstance(author, dict) else str(author)
        if "AI Agent" in name:
            violations.append((Path("pyproject.toml"), 0, f"Placeholder author: '{name}'"))
    return violations


def check_not_implemented(files: list[Path]) -> list[tuple[Path, int, str]]:
    """Flag NotImplementedError outside abstract methods."""
    violations: list[tuple[Path, int, str]] = []
    for path in files:
        if "src" not in path.parts or path.suffix != ".py":
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, PermissionError):
            continue
        in_abstract = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "@abstractmethod" in stripped:
                in_abstract = True
                continue
            if "raise NotImplementedError" in stripped and not in_abstract:
                violations.append(
                    (
                        path.relative_to(ROOT),
                        i,
                        "NotImplementedError in non-abstract method",
                    )
                )
            if stripped and not stripped.startswith("@") and "def " not in stripped:
                in_abstract = False
    return violations


def check_python_version() -> list[tuple[Path, int, str]]:
    """Verify pyproject requires-python matches ruff target and mypy version."""
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return []
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    requires = data.get("project", {}).get("requires-python", "")
    ruff_target = data.get("tool", {}).get("ruff", {}).get("target-version", "")
    mypy_version = data.get("tool", {}).get("mypy", {}).get("python_version", "")

    match = re.search(r"(\d+)\.(\d+)", requires)
    if not match:
        return []
    expected_ruff = f"py{match.group(1)}{match.group(2)}"
    expected_mypy = f"{match.group(1)}.{match.group(2)}"

    out: list[tuple[Path, int, str]] = []
    if ruff_target and ruff_target != expected_ruff:
        out.append(
            (
                Path("pyproject.toml"),
                0,
                f"requires-python={requires} but ruff target-version={ruff_target}",
            )
        )
    if mypy_version and mypy_version != expected_mypy:
        out.append(
            (
                Path("pyproject.toml"),
                0,
                f"requires-python={requires} but mypy python_version={mypy_version}",
            )
        )
    return out


def main() -> int:
    files = find_code_files(ROOT)
    all_violations: list[tuple[Path, int, str]] = []
    all_violations.extend(check_forbidden(files))
    all_violations.extend(check_pyproject_authors())
    all_violations.extend(check_not_implemented(files))
    all_violations.extend(check_python_version())

    if all_violations:
        print(f"FAIL: {len(all_violations)} anti-pattern violation(s):")
        for path, line_no, description in all_violations:
            location = f"{path}:{line_no}" if line_no > 0 else str(path)
            print(f"  {location}  {description}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
