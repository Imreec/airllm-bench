"""Build the right Runner from its name + setup config (T5.4).

Centralizes the constructor-injection each runner needs (device / shards path /
gguf path / n_gpu_layers, all from ``setup.json``) so the orchestrator and the
keyless mock smoke test share one construction path. The heavy backends are still
imported lazily *inside* the runners, so building any runner here stays keyless.
"""

from __future__ import annotations

from airllm_bench.runners.mock import MockRunner
from airllm_bench.runners.protocol import Runner
from airllm_bench.shared.config_models import SetupConfig


def build_runner(name: str, setup: SetupConfig) -> Runner:
    """Construct the runner named ``name`` with params drawn from ``setup``.

    Raises:
        ValueError: on an unknown runner name, or a missing llamacpp ``gguf_path``.
    """
    repo_id = setup.model.repo_id
    opts = setup.runners.get(name, {})
    if name == "mock":
        return MockRunner()
    if name == "baseline_hf":
        from airllm_bench.runners.baseline_hf import BaselineHFRunner

        return BaselineHFRunner(repo_id, device=opts.get("device", 0))
    if name == "airllm":
        from airllm_bench.runners.airllm import AirLLMRunner

        return AirLLMRunner(
            repo_id, setup.paths.layer_shards_saving_path, device=opts.get("device", "cuda")
        )
    if name == "llamacpp":
        from airllm_bench.runners.llamacpp import LlamaCppRunner

        gguf_path = opts.get("gguf_path")
        if not gguf_path:
            msg = "llamacpp runner needs runners.llamacpp.gguf_path in setup.json"
            raise ValueError(msg)
        return LlamaCppRunner(gguf_path, n_gpu_layers=opts.get("n_gpu_layers", 0))
    msg = f"unknown runner: {name!r}"
    raise ValueError(msg)
