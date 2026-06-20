"""Environment metadata captured into each ``RunResult.env`` (PLAN §4).

os / python / torch / cuda / driver, so a committed result is self-describing and the
report can cite the exact stack behind every number. ``torch`` is runtime-only and NVML
needs a GPU, so both are probed defensively — a missing piece yields ``""`` rather than
failing a run (metadata must never break a measurement).
"""

from __future__ import annotations

import platform


def _torch_info() -> dict[str, str]:
    """torch/cuda versions, or empty strings when the runtime stack isn't installed."""
    try:
        import torch
    except Exception:  # noqa: BLE001 — best-effort metadata; never fail a run for it
        return {"torch": "", "cuda": "", "cuda_available": ""}
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "cuda_available": str(torch.cuda.is_available()),
    }


def _driver_info() -> str:
    """NVIDIA driver version via NVML, or "" when no GPU/driver is present."""
    try:
        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
    except Exception:  # noqa: BLE001 — best-effort metadata
        return ""
    return version.decode() if isinstance(version, bytes) else str(version)


def collect_env() -> dict[str, str]:
    """Best-effort stack metadata; missing pieces are empty strings, never errors."""
    env = {"os": platform.platform(), "python": platform.python_version()}
    env.update(_torch_info())
    env["driver"] = _driver_info()
    return env
