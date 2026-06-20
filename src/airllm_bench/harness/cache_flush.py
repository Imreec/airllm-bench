"""Cold-cache enforcement so the cold/warm contrast is honest (T5.4, ADR 0001).

A true *cold* run needs an empty OS page cache. On Windows that means flushing the
Standby List (Sysinternals ``RAMMap -Et``), which requires Administrator — a silent
Access-Denied would no-op and corrupt "cold" data, so the privilege check is FAIL-LOUD,
and the flush tool itself must exit 0 (``check=True``).

We deliberately do **not** gate on a freed-memory magnitude. On Windows the Standby List
already counts toward ``available`` memory, so emptying it barely moves ``available``
(measured ~30 MB even with GBs cached) — an ``available``-rise check can never pass here,
which is the quirk ADR 0001 flagged. Cold-ness is evidenced by the cold/warm *timing*
delta instead; a broken flush would show cold ≈ warm. The OS-specific bits (admin check,
the flush subprocess) are injectable, so CI tests this without elevation or the tool.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


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


def require_admin(is_admin: Callable[[], bool] = _windows_is_admin) -> None:
    """Fail loud if not elevated — a non-admin flush silently no-ops and corrupts cold data."""
    if not is_admin():
        msg = (
            "cold-cache flush needs Administrator (RAMMap -Et); "
            "re-run from an elevated shell, or pass --no-flush for warm-only scenarios."
        )
        raise NotElevatedError(msg)


def flush_standby_list(
    command: Sequence[str], *, run: Callable[..., object] = subprocess.run
) -> None:
    """Invoke the standby-list flush tool (e.g. ``RAMMap64.exe -accepteula -Et``); must exit 0."""
    run(list(command), check=True)


def build_cold_flush(
    cfg: Mapping[str, Any],
    *,
    is_admin: Callable[[], bool] = _windows_is_admin,
    run: Callable[..., object] = subprocess.run,
) -> Callable[[], None]:
    """Compose the cold-flush step from config: require admin, then empty the standby list."""
    command = cfg["command"]

    def cold_flush() -> None:
        require_admin(is_admin)
        flush_standby_list(command, run=run)

    return cold_flush
