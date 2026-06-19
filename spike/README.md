# G-SPIKE — go/no-go (throwaway)

**Purpose:** prove the stack runs on *this* hardware before we lock the model and write any harness
code (TODO Phase 2, PRD D1). These scripts are **throwaway** — they validate, they don't ship. The
durable output is a go/no-go ADR written from the numbers you paste back.

> **Run these on the GPU box, native Windows first.** Nothing here runs in CI. Run from an
> **Administrator** terminal is not required for the spike (only the later matrix flush needs it).

## Setup (once)

```bash
make install-runtime        # installs torch/transformers/accelerate/airllm/bitsandbytes (NOT llama.cpp)
```
If `make install-runtime` fails, **stop and paste the error** — that's R1 (the Windows stack risk)
firing, and it's exactly what the spike exists to surface. We decide native-vs-WSL2 from that.

Point AirLLM's layer shards at your **NVMe** (never the slow HDD). Set this before script 03:
```bash
# PowerShell: $env:AIRLLM_SHARDS = "D:\airllm_shards"   (use YOUR NVMe drive letter)
# Git Bash:   export AIRLLM_SHARDS="/d/airllm_shards"
```

## Run order (and what to paste back)

```bash
uv run --extra runtime python spike/01_stack_check.py
uv run --extra runtime python spike/02_harness_proof.py
uv run --extra runtime python spike/03_airllm_32b.py --compression 4bit
# then, if 4bit worked and was tolerable, also:
uv run --extra runtime python spike/03_airllm_32b.py --compression none
```

| Script | Proves (TODO) | Paste back |
|--------|---------------|------------|
| `01_stack_check.py` | T2.1 — stack imports, CUDA sees the 3080 Ti, bitsandbytes 4bit works | the whole printed block |
| `02_harness_proof.py` | T2.2 + T2.5 — streaming TTFT/ITL, NVML power/VRAM, psutil RAM, logits | the SUMMARY block |
| `03_airllm_32b.py` | T2.3 + T2.4 + T2.5 — one 32B token timed, cold-vs-warm page cache, AirLLM logits | the SUMMARY block |

**The single most important number** is `03`'s measured **seconds-per-token** for 32B — it sizes the
whole experiment matrix (`max_new_tokens`, prompt count, reps). The cold-vs-warm gap tells us whether
NF4 lives in the OS page cache.

> ⚠️ `03_airllm_32b.py` is **slow** — it downloads ~tens of GB the first time and each token streams
> all 64 layers from disk. Expect minutes. If it hangs for hours, Ctrl-C, paste the last output, and
> we replan (brief's tip: stop, feed the error back, don't wait it out).
