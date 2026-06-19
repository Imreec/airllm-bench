# ADR 0001 — G-SPIKE go/no-go: environment locked, model confirmed

- **Status:** Accepted
- **Date:** 2026-06-19
- **Context:** PRD D1 (mandatory gated spike) · TODO Phase 2 (T2.1–T2.5) · resolves PRD §7 open items.

## Context

Before writing any harness code we ran a throwaway spike (`spike/`) to prove the stack runs on
*this* hardware and to measure the one number that sizes the whole experiment: seconds-per-token for
the 32B model through AirLLM. The three deferred decisions were the model (32B vs 14B), the matrix
size, and native-Windows vs WSL2.

## Decision

**GO, on native Windows.** Lock **Qwen2.5-32B-Instruct**. No WSL2.

### Verified (all GO)
| Check | Result |
|-------|--------|
| T2.1 stack on native Windows | torch `2.6.0+cu124` sees the RTX 3080 Ti (12 GB, cc 8.6); bitsandbytes 4-bit forward works; airllm imports |
| T2.2 instrumentation | streaming TTFT/TPOT split, NVML power+VRAM, psutil RAM — all captured on one clock |
| T2.3 32B runs via AirLLM (4-bit) | **~20 s/token** (warm); load-from-shards 1.7 s; peak VRAM 6.7 GB; peak power 157 W (NVML) |
| T2.4 page cache | genuine cold/warm = **1.84** (18 GB 4-bit shards fit in 32 GB RAM); a *pre-warmed* re-run gave 0.99 — proving a true cold run **requires an explicit cache flush** |
| T2.5 logits / perplexity | AirLLM forward returns logits as a tuple → `out[0]`, shape `(1, n, 152064)`. **Perplexity computable** (risk R5 cleared) |

### Locked dependency constellation (hard-won — each pin has a reason)
- **torch from the `cu124` index** — PyPI serves CPU-only wheels on Windows (`torch==*+cpu`,
  `cuda_available=False`), which would defeat the experiment.
- **transformers `>=4.40,<4.43`** — 4.43 moved RoPE `position_embeddings` to the model level; AirLLM
  drives layers itself with the older self-contained-rotary API, so ≥4.43 crashes mid-generation
  (`cos, sin = position_embeddings` on `None`). 4.42.x supports Qwen2.5 and satisfies the next cap.
- **optimum `<2`** + **transformers `<4.49`** — airllm imports `optimum.bettertransformer`, removed in
  optimum 2.0 and capped at transformers <4.49.
- **sentencepiece + protobuf** — airllm tokenizer deps, not pulled transitively.
- runtime/llamacpp extras split so the build-fragile `llama-cpp-python` can't block the AirLLM path.

### Matrix sizing (consequence)
- **4-bit 32B ≈ 20 s/token** → a 10-token generation ≈ 3.3 min/scenario. Tractable; afford ~10–20
  output tokens at 4-bit.
- **FP16 (62 GB shards, does NOT fit 32 GB RAM)** → expect disk-streaming every layer; the cold first
  pass measured ~7 s/layer × 67 ≈ **~8 min/token**. Keep FP16 scenarios to **a handful of tokens**
  and few reps. (FP16/Q8 timing to be measured for real in Phase 5.)
- This validates the experiment's thesis: 4-bit fits RAM (memory-hierarchy win), FP16 does not
  (capacity wall → I/O bound).

### Disk plan (corrects an earlier optimistic assumption)
AirLLM writes **compression-specific shards** (4-bit set = 18 GB), **not** a reusable FP16 set. So
FP16/Q8/Q4 each write their own shards (≈62/31/18 GB) on top of the ~62 GB HF download — ~173 GB if
accumulated, against ~195 GB free on C:. A near-full OS NVMe risks a hard system crash (not a clean
failure), so **the Phase-5 orchestrator MUST NOT accumulate all shard sets on C: at once** (PR #5
cross-model review).

**Mandatory Phase-5 disk policy:**
1. **`HF_HOME` → D:** (the HDD) — the original safetensors are read only once per compression level
   during sharding, so the slow drive is fine there; frees ~62 GB on the NVMe.
2. **One shard set on C: at a time** — create → run → **delete** that compression level's shards
   before sharding the next. Shards are regenerable from the HF cache; the committed **evidence is the
   results JSON**, not the shards. So C: never holds more than ~62 GB (one FP16 set).
3. **Baseline headroom + correct load path** — the `baseline_hf` FP16 OOM run must load **GPU-only via
   `device_map={"":0}` with `low_cpu_mem_usage=True`** (NOT `.to("cuda")`, which would stage all 64 GB
   in 32 GB host RAM → pagefile). GPU-only makes it OOM on VRAM cleanly without a host-RAM/pagefile
   spill; with policy (1)+(2) C: also stays far from full, so even a residual pagefile bump can't
   suffocate the drive.

## Consequences / follow-ups

- The dependency pins move from the spike branch into `pyproject.toml` on `main` (they are
  load-bearing, not throwaway).
- **`KNOWN_LIMITATIONS.md` candidates surfaced by the spike:** (a) Windows `psutil` exposes no system
  *cached*/standby figure → the page-cache signal is cold-vs-warm timing + `available`, not a cache
  number (revises PRD D5); (b) a true cold run needs an admin-privileged standby-list flush
  (confirms TODO T5.4); (c) the transformers `<4.43` pin is an AirLLM constraint, not a free choice.
- The real `airllm` runner must **pre-create the shards dir** (AirLLM's `check_space` requires it) and
  **extract logits from the forward tuple**.

## Alternatives considered

- **WSL2** — rejected: native Windows works, and WSL2 would add a VHDX/9p I/O layer plus the ~50%
  RAM cap that would distort the very page-cache effect we measured (cold/warm 1.84).
- **Drop to 14B** — not needed: 32B runs at a tractable ~20 s/token at 4-bit.
- **transformers ≥4.43 / optimum 2.x** — rejected: incompatible with airllm's layer driver.
