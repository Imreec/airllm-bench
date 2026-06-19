"""Versioned, fail-loud config loading shared by every config file.

Carries the proven HW2 pattern: parse JSON, fail loudly on a version mismatch
(prefix check, so ``1.00``/``1.01`` are both accepted), and keep the untouched
payload under ``raw_data`` as an escape hatch. No hardcoded model strings,
prices, or rates live in code — they all flow through here from ``config/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

#: Every config file's ``version`` must start with this prefix.
CONFIG_VERSION_PREFIX = "1."


class ConfigVersionError(ValueError):
    """Raised when a config file's ``version`` is absent or unsupported."""


def load_versioned[T: BaseModel](path: str | Path, model_cls: type[T]) -> T:
    """Load JSON at ``path``, check its version prefix, validate into ``model_cls``.

    The whole payload is preserved under the model's ``raw_data`` field.

    Raises:
        FileNotFoundError: when ``path`` does not exist.
        ConfigVersionError: when ``version`` is missing or not ``1.*``.
    """
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, str) or not version.startswith(CONFIG_VERSION_PREFIX):
        msg = f"config version {version!r} must start with {CONFIG_VERSION_PREFIX!r}"
        raise ConfigVersionError(msg)
    return model_cls.model_validate({**data, "raw_data": data})
