# airllm-bench

Benchmarking a **massive LLM run locally** via **AirLLM + quantization** on consumer hardware
(RTX 3080 Ti, 12 GB VRAM / 32 GB RAM), and analyzing the cost/benefit — performance, economics, and
the memory-hierarchy physics behind it.

> **Status: bootstrapping.** Planning is complete and approved; the build has just begun. This README
> is a placeholder — **it is not yet the technical report.** The full report (the graded deliverable)
> is written in Phase 7, with all measured results, tables, and figures embedded inline. Nothing below
> claims results that don't exist yet.

## What this will be

A reproducible experiment that takes a model too large for the GPU (Qwen2.5-32B, FP16 ≈ 64 GB), shows
the direct baseline OOMs, then runs it via AirLLM layer-streaming + quantization — and asks
**"when does layer-by-layer streaming actually pay off?"** vs. the realistic alternative
(llama.cpp Q4 with GPU offload). Outputs: TTFT/TPOT/throughput/peak-memory/power, an On-Prem-vs-API
break-even analysis, and a hierarchical roofline tying the measurements to theory.

## Planning documents

| Doc | Purpose |
|-----|---------|
| [docs/PRD.md](docs/PRD.md) | Requirements + the 13 locked decisions |
| [docs/PLAN.md](docs/PLAN.md) | Architecture, the runner/RunResult contracts |
| [docs/TODO.md](docs/TODO.md) | Atomic task tracker (7 phases, the G-SPIKE gate) |

## Development

```bash
make install      # core + dev deps (keyless, Tier-1 analysis stack)
make grade        # full quality gate (ruff + mypy + tests + scanners)
```

The heavy model-runtime stack (torch/airllm/bitsandbytes) is hardware-bound and installed separately
with `make install-runtime` on the GPU box. See [CLAUDE.md](CLAUDE.md) for standards.

## License

MIT — see [LICENSE](LICENSE). Authors: [AUTHORS.md](AUTHORS.md).
