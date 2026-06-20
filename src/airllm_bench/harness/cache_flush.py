"""Cold-cache enforcement so the cold/warm contrast is honest (T5.4, ADR 0001).

A true *cold* run needs an empty OS page cache. On Windows that means flushing the
Standby List (EmptyStandbyList/RAMMap), which requires Administrator — a silent
Access-Denied would no-op and corrupt "cold" data, so the privilege check is
FAIL-LOUD. After flushing we verify available memory actually rose (belt-and-
suspenders). Every OS-specific bit (admin check, the flush subprocess, the memory
read) is injectable, so CI tests this without elevation or the external tool.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_MB = 1024 * 1024


class NotElevatedError(PermissionError):
    """Raised when a cold-cache flush is attempted without Administrator."""


def _windows_is_admin() -> bool:  # pragma: no cover — Windows-elevation-only, never runs in CI
    # ctypes.windll is win32-only. Keeping it INSIDE the `== "win32"` block (not a
    # fall-through after an early return) is what mypy exempts on both OSes: on Linux
    # the branch is skipped (no attr-defined, no unreachable); on Windows the `else` is
    # the guard's non-matching branch (also exempt). The explicit else is required for
    # that exemption, so RET505 is suppressed deliberately.
    if sys.platform == "win32":
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    else:  # noqa: RET505 — explicit else keeps mypy's platform-guard narrowing clean
        return False


def _psutil_avail_mb() -> float:  # pragma: no cover — trivial hardware read, exercised on the box
    import psutil

    return psutil.virtual_memory().available / _MB


def require_admin(is_admin: Callable[[], bool] = _windows_is_admin) -> None:
    """Fail loud if not elevated — a non-admin flush silently no-ops and corrupts cold data."""
    if not is_admin():
        msg = (
            "cold-cache flush needs Administrator (EmptyStandbyList); "
            "re-run from an elevated shell, or pass --no-flush for warm-only scenarios."
        )
        raise NotElevatedError(msg)


def flush_standby_list(
    command: Sequence[str], *, run: Callable[..., object] = subprocess.run
) -> None:
    """Invoke the standby-list flush tool (e.g. ``EmptyStandbyList.exe standbylist``)."""
    run(list(command), check=True)


def verify_cache_dropped(before_mb: float, after_mb: float, *, min_rise_mb: float) -> bool:
    """Flushing standby frees cached pages, so available memory should RISE by ``min_rise_mb``."""
    return (after_mb - before_mb) >= min_rise_mb


def build_cold_flush(
    cfg: Mapping[str, Any],
    *,
    is_admin: Callable[[], bool] = _windows_is_admin,
    run: Callable[..., object] = subprocess.run,
    read_avail_mb: Callable[[], float] = _psutil_avail_mb,
) -> Callable[[], None]:
    """Compose the full cold-flush step from config: require-admin → flush → verify."""
    command = cfg["command"]
    min_rise = float(cfg.get("min_rise_mb", 0))

    def cold_flush() -> None:
        require_admin(is_admin)
        before = read_avail_mb()
        flush_standby_list(command, run=run)
        after = read_avail_mb()
        if not verify_cache_dropped(before, after, min_rise_mb=min_rise):
            msg = f"standby flush freed only {after - before:.0f} MB (< {min_rise:.0f} required)"
            raise RuntimeError(msg)

    return cold_flush
