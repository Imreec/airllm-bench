# Tier-2 Runbook — executing the measured matrix (T5.5)

The Tier-2 runs are hardware-bound and **not** in CI. This is the exact procedure to
reproduce the committed `results/` JSON on the documented box (RTX 3080 Ti / 32 GB / Ryzen 9).
Everything here is run **on the machine, by the operator** — Claude prepares it; the operator
launches the elevated batches.

## Hardware / disk facts that shape the procedure (ADR 0001)

- **C: = NVMe, D: = HDD.** Anything streamed *during inference* must be on C:.
  - AirLLM **shards** stream every forward pass → **C:** (`C:/airllm_shards`).
  - The llama.cpp **GGUF** is mmap-read during inference → **C:** (`C:/models`).
  - The HF **safetensors cache** is read only *once per compression level during sharding* →
    **D:** (`D:/hf_cache`, HDD is fine here).
- **AirLLM shards are compression-specific** (≈18 / 31 / 62 GB for 4-bit / 8-bit / FP16) and are
  regenerable from the HF cache. The committed **evidence is the results JSON, not the shards**.
  C: must never hold more than **one** big shard set at a time → delete between compression levels.
- **A true cold run needs an empty OS page cache.** Flushing the Windows Standby List requires
  **Administrator**; the orchestrator fails loud (`NotElevatedError`) otherwise, because a silent
  Access-Denied would corrupt "cold" data. So **cold batches run from an elevated PowerShell**.
- **Sharding must happen *before* a measured cold run** — the first load of a compression level
  writes its shards (minutes of disk I/O). Pre-shard once (a throwaway load), then measure.

## Prerequisites

1. `uv sync --extra runtime` (torch `cu124` + airllm + transformers `<4.43`). For the competitor also
   `uv sync --extra runtime --extra llamacpp` (the GGUF runner).
2. **`C:/tools/RAMMap64.exe`** present (Sysinternals; `-Et` empties the Standby List) — the cold-cache
   tool named in `config/setup.json → cold_flush.command`. The first `-accepteula` run is
   non-interactive thereafter. Without it, run warm-only (`--no-flush`).
3. **GGUF** `C:/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf` (~18 GB) for the llama.cpp batch.
4. HF cache relocated to D: (`setx HF_HOME D:/hf_cache`, then move the existing cache) before the
   8-bit/FP16 sharding steps — frees C: for the FP16 shard set.

## Disk choreography (one shard set on C: at a time)

| Step | Compression | Shards on C: | Action |
|------|-------------|--------------|--------|
| A | 4-bit (already sharded, 18 GB) | `splitted_model.4bit` | run; keep |
| B | — | — | move HF cache C:→D:; (optionally delete 4-bit shards) |
| C | — (GGUF) | + GGUF 18 GB | download GGUF to C:; run llama.cpp |
| D | 8-bit | + `splitted_model.8bit` 31 GB | pre-shard → run → **delete 8-bit shards** |
| E | FP16 | + `splitted_model` 62 GB | pre-shard → run → **delete FP16 shards** |

## Execution

Each batch is one `bench` invocation over an ordered scenario list (cold scenarios first so the
flush precedes them; the paired warm runs reuse the now-populated cache). From an **elevated**
PowerShell at the repo root:

```powershell
# Batch A — AirLLM 4-bit (headline cold/warm + warm rep + TTFT-vs-length sweep)
uv run airllm-bench bench --experiments `
  config/experiments/airllm-nf4-cold.json `
  config/experiments/airllm-nf4-warm.json `
  config/experiments/airllm-nf4-warm-r2.json `
  config/experiments/airllm-nf4-len64.json `
  config/experiments/airllm-nf4-len256.json   # length sweep caps at 256 (AirLLM max_seq_len=512, L-08)

# Batch C — llama.cpp competitor (warm + rep); no cold here → --no-flush is fine
uv run airllm-bench bench --no-flush --experiments `
  config/experiments/llamacpp-q4-warm.json `
  config/experiments/llamacpp-q4-warm-r2.json

# Batch D — AirLLM 8-bit.
#   D1. PRE-SHARD (non-admin, one-time, writes splitted_model.8bit). MUST precede the cold run,
#       else sharding warms the cache and the "cold" run isn't cold (the batch's flush then evicts
#       the freshly-written shards so the cold run reads them genuinely cold):
uv run python -c "from airllm_bench.runners.factory import build_runner; from airllm_bench.shared.config_models import SetupConfig, ExperimentConfig; s=SetupConfig.from_file('config/setup.json'); e=ExperimentConfig.from_file('config/experiments/airllm-int8-cold.json'); build_runner('airllm', s).load(e)"
#   D2. Measure (elevated):
uv run airllm-bench bench --experiments `
  config/experiments/airllm-int8-cold.json `
  config/experiments/airllm-int8-warm.json
#   D3. Free C: for FP16:  Remove-Item -Recurse -Force C:/airllm_shards/splitted_model.8bit

# Batch E — AirLLM FP16 (~70 min, the long pole) + baseline OOM.
#   E1. PRE-SHARD (writes splitted_model, ~62 GB):
uv run python -c "from airllm_bench.runners.factory import build_runner; from airllm_bench.shared.config_models import SetupConfig, ExperimentConfig; s=SetupConfig.from_file('config/setup.json'); e=ExperimentConfig.from_file('config/experiments/airllm-none-cold.json'); build_runner('airllm', s).load(e)"
#   E2. Measure (elevated):
uv run airllm-bench bench --experiments `
  config/experiments/airllm-none-cold.json `
  config/experiments/airllm-none-warm.json `
  config/experiments/baseline-oom.json
#   E3. Reclaim C:  Remove-Item -Recurse -Force C:/airllm_shards/splitted_model
```

> **Pre-sharding** applies to 8-bit/FP16 only; the 4-bit shards already exist from the G-SPIKE, so
> Batch A's flush evicts them and measures a genuine cold read with no pre-shard step.

Each scenario runs in its own subprocess and **appends one `RunResult` line** to
`results/<exp_id>.jsonl`. After all batches: commit `results/*.jsonl` as the Tier-2 evidence
(T5.5), then Phase 6 (`economics`/`roofline`/`plotting`) consumes them keylessly.

## Estimated wall-clock (ADR 0001 rates)

4-bit batch ≈ 20 min · llama.cpp ≈ a few min · 8-bit ≈ 15 min · **FP16 ≈ 70 min (long pole)** ·
baseline OOM ≈ seconds. Plus setup (HF move ≈ 10 min, GGUF download, per-level sharding). ~3–3.5 h total.
