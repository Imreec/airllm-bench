# Extending the Benchmark

The core stays untouched; everything below is data or a small adapter. This is the engineering-flair
surface: adding a model, a quant level, or a scenario should never mean editing the harness.

## Add a model
Edit `config/setup.json` → `model.repo_id` + `params`. The runners load it via `SetupConfig`; no code
changes. (Mind the disk policy in ADR 0001 — shards are large.)

## Add a quantization level
- **AirLLM path:** set `quant` in the experiment config to a bitsandbytes mode (`none` | `int8` | `nf4`).
- **llama.cpp path:** a GGUF K-quant (`q4_k_m` | `q8_0`).
Remember (ADR 0001): AirLLM shards are compression-specific, so each level writes its own shards —
the orchestrator creates → runs → deletes one set at a time.

## Add a scenario
Drop a new `config/experiments/<id>.json` (one scenario: runner, quant, prompt, `max_new_tokens`,
`phase`). The matrix orchestrator (Phase 5) expands the length × cold/warm sweep.

## Add a runner
Implement the `Runner` Protocol (`load` / `stream` / `logits` / `unload`) in `runners/` and register
it. The harness measures any runner identically.

## Future extensions (out of scope, D9)
- **QLoRA fine-tune** — connect inference to training (brief's example). Deferred; one strong
  extension (the hierarchical roofline) beats two half-built ones.
- Multi-model size comparison — explicitly *not* the primary extension (flattest option).
