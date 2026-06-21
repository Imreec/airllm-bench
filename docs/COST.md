# Cost Ledger

## Live API spend: $0.00

HW5 is **keyless and makes no paid API calls** (D8, [ADR 0002](adr/0002-gatekeeper-na.md)). The "API
cost" in this project is a *modeled* quantity — paper math on token counts against documented price
constants — not real spend. There is no attributed-vs-dashboard reconciliation to do here, unlike a
live-LLM project.

## Incidental costs (non-USD)
- **Model download:** the base FP16 safetensors (~62 GB) over the HF Hub, once (bandwidth, not billed).
  AirLLM re-shards it per compression level on disk; the committed evidence is the `results/` JSON, not
  the weights (D13 — weights are git-ignored).
- **Electricity:** the Tier-2 benchmark runs draw real power — **measured** (GPU via NVML, integrated
  over each run) and **estimated** (CPU via TDP, [L-03](KNOWN_LIMITATIONS.md)). This measured energy is
  what drives the on-prem economic line, not a nameplate guess.

## Modeled costs (the analysis)
The full On-Prem vs API vs Cloud break-even — every assumption parametrized in
[`config/economics.json`](../config/economics.json), each constant carrying a source + retrieval date —
is in the [README economics section](../README.md#-economics--when-does-it-pay-for-itself). The
headline modeled figures (per 1 M output tokens):

| Line | Modeled $/1M output tok |
|------|------------------------:|
| API (OpenRouter Qwen2.5-Coder-32B) | $1.66 |
| API + caching | $1.36 |
| On-Prem realistic (llama.cpp, + $1200 CAPEX) | $4.30 marginal |
| On-Prem AirLLM (cautionary) | $189.92 marginal |
| Cloud GPU (RunPod RTX 3090) | $67.25 marginal |

**Finding:** the realistic on-prem *marginal electricity cost alone* ($4.30/Mtok) exceeds the API's
*total* price ($1.66/Mtok), so single-stream on-prem at this hardware class **never breaks even** — a
clean, measured negative result. The quality-parity and SKU-comparability caveats are in
[L-09](KNOWN_LIMITATIONS.md).

This file exists to make the **zero live spend** explicit and to point at where the modeled costs live.
