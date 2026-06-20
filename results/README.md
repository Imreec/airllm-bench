# Tier-2 measured results (T5.5)

Raw `RunResult` JSON, one scenario per file (one line each), measured on the documented box and
committed as **evidence** (D11). Every downstream figure/number (Phase 6–7) is a pure function of
these files — they are the source of truth. Regenerate via `docs/RUNBOOK_TIER2.md`.

**Environment** (captured in every row's `env`): Windows 10 19045 · Python 3.12 · torch 2.6.0+cu124 ·
CUDA 12.4 · driver 591.74 · RTX 3080 Ti (12 GB) · Ryzen 9 5900X · 32 GB RAM. Greedy (temperature 0);
perplexity from a teacher-forced forward pass on the 40-token fixed prompt.

## The matrix

| file | runner | quant | phase | headline |
|------|--------|-------|-------|----------|
| `airllm-nf4-cold` / `-warm` / `-warm-r2` | AirLLM | 4-bit (18 GB) | cold/warm | TTFT 43.7 → 19.3 s — **2.26× warm win** (fits 32 GB RAM) |
| `airllm-nf4-len64` / `-len256` | AirLLM | 4-bit | warm | TTFT ~flat vs length → **streaming-bound** prefill |
| `airllm-int8-cold` / `-warm` | AirLLM | 8-bit (31 GB) | cold/warm | 54.3 → 50.8 s — **~1.07× (no real win)**, at the RAM edge |
| `airllm-none-cold` / `-warm` | AirLLM | FP16 (61 GB) | cold/warm | 107.4 → 105.5 s — **~1.02× (no win)**, exceeds RAM |
| `baseline-oom` | baseline_hf | FP16 | — | clean VRAM OOM at load, peak VRAM 12.16 GB (capacity wall, D3) |
| `llamacpp-q4-warm` / `-warm-r2` | llama.cpp | Q4_K_M | warm | the realistic GPU-offload competitor (TPOT ~0.46 s) |

## What the data shows

- **Memory hierarchy:** the warm (page-cache) speedup collapses as the model outgrows RAM —
  2.26× (4-bit, fits) → 1.07× (8-bit, edge) → 1.02× (FP16, won't fit). This is the project thesis.
- **Speed vs bits:** TPOT 20 → 52 → 108 s/token as precision rises (more bytes streamed per token).
- **Quality vs bits:** perplexity 39.9 (4-bit) → 24.2 (8-bit) → 23.7 (FP16) — 8-bit is near-lossless,
  4-bit trades quality for speed.
- **FP16 was ~1.8 min/token**, far better than ADR 0001's ~8 min/token estimate (revises that guess).
