"""airllm-bench: local massive-LLM benchmarking via AirLLM + quantization.

Package skeleton. Modules (harness, runners, metrics, economics, roofline, plotting,
shared, cli) land in later TODO phases. Heavy model-runtime imports are deferred to the
runner modules so the keyless Tier-1 analysis stack imports without the GPU stack present.
"""

from __future__ import annotations

__version__ = "1.00"
__all__ = ["__version__"]
