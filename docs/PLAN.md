# PLAN — Architecture & Contracts (`airllm-bench`)

**Status:** Draft — awaiting PLAN-gate approval. Derives from `docs/PRD.md` (D1–D13).
**Scope:** how the system is built — modules, the runner contract, the harness, data shapes,
config schemas, CI/testing. Not *what* to do step-by-step (that's `TODO.md`).

Design rule throughout: **measurement is separated from analysis is separated from presentation.**
Raw JSON is the source of truth; everything downstream is a pure function of committed data.

---

## 1. C4 — Context & Containers

### Context
One operator (Imree, driving Claude Code) runs the benchmark on the documented Windows box. No
external services at runtime (D8: keyless). Inputs = config files + a local HF model cache. Outputs
= raw JSON under `results/`, figures under `figures/`, and the report (`README.md`).

### Containers (logical)
```
config/ (data)  ──►  SDK functions  ──►  results/*.jsonl (truth)  ──►  figures/  ──►  README.md
                         │                                              ▲
                         ├─ harness + runners (Tier 2: needs hardware)  │
                         └─ metrics / economics / roofline (Tier 1: keyless, offline)
```
- **Tier 2 (hardware-bound):** `harness` + `runners` produce raw data on the real GPU/box.
- **Tier 1 (keyless, offline, CI):** `metrics`, `economics`, `roofline`, plotting — pure functions
  of committed JSON; regenerate every figure with no GPU/key/network (D11).

---

## 2. Module layout (`src/airllm_bench/`)

```
src/airllm_bench/
├── shared/          # config loader (versioned, fail-loud), config_models, JSONL StructuredLogger
├── harness/         # the ONE measurement harness + the background ResourceSampler thread
├── runners/         # runner Protocol + baseline_hf / airllm / llamacpp implementations
├── metrics/         # TTFT/TPOT/throughput/perplexity derivation from raw RunResult
├── economics/       # costing (API token-cost, on-prem CAPEX+OPEX), break-even
├── roofline/        # ceilings (compute/HBM/PCIe/NVMe) + operating-point placement
├── plotting/        # figures generated FROM results/ JSON (never inline)
└── cli.py           # thin CLI: bench / figures / economics / report-check
```
Plus repo-root `scripts/` (CI checkers) and `config/`, `results/`, `figures/`, `experiments/`,
`reports/`. **Every file ≤ 150 lines** (CI-enforced); split modules before they grow.

No SDK-facade layer (D9) — `cli.py` calls plain module functions directly.

---

## 3. The runner contract (the load-bearing interface)

All three runtimes implement one `Protocol` so the harness measures them identically (D3, D4).

```python
class Runner(Protocol):
    name: str
    def load(self, cfg: ExperimentConfig) -> None: ...          # may raise on OOM (baseline)
    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]: ...
    def logits(self, text: str) -> LogitsResult: ...            # native-tokenize + one forward pass
    def unload(self) -> None: ...
```

- `TokenEvent` = `{token_id, text, t_monotonic}` — emitted per token so the harness timestamps
  TTFT (first event) and the ITL series (gaps between events). **No total-time ÷ tokens anywhere.**
- `stream()` is the only generation path; greedy/temperature-0 fixed in the runner (D4).
- `logits(text)` does its **own native tokenization** and returns `LogitsResult{token_ids, logits}`
  (D10). Passing HF `input_ids` to the llama.cpp runner would assume HF↔GGUF tokenizer parity and
  silently corrupt its perplexity, so each runtime tokenizes itself. **Cross-runtime** perplexity
  comparison is only valid when token counts match on the eval text (Qwen HF vs GGUF align closely;
  any divergence is disclosed). Spike (D1e) verifies AirLLM exposes logits.
- **`ExperimentConfig` describes exactly ONE scenario** — one prompt, one prompt-length, one `phase`
  (cold | warm). The length × cold/warm matrix is orchestrated *outside* the harness (§4).

**Implementations:**
| Runner | Loads via | Quant | Notes |
|--------|-----------|-------|-------|
| `baseline_hf` | `transformers`, `device_map={"":0}` / `.to("cuda")` | FP16 | **Never** `device_map="auto"` — must OOM clean at load (D3). `load()` is expected to raise. |
| `airllm` | `AirLLMModel` (AutoModel path) | FP16 / INT8 / NF4 via `compression` | `layer_shards_saving_path` → **NVMe** (never HDD). |
| `llamacpp` | Ollama / `llama-cpp-python` GGUF | Q4_K_M (+Q8) | partial GPU offload (`n_gpu_layers`). The realistic competitor. |

A 4th `mock` runner (canned `TokenEvent`s) exists for the keyless CI harness-wiring smoke test (D11).

---

## 4. The harness + resource sampler

`harness.run(runner, cfg) -> RunResult` executes **exactly one scenario** (one prompt-length, one
`phase`) and emits **one** `RunResult` — so the harness and the scalar schema align (D4, D5):

1. Start a **`ResourceSampler`** background thread sampling on one monotonic clock:
   NVML GPU power (mW) + NVML VRAM used, psutil process RSS, psutil system used/cached.
2. `runner.load(cfg)` — wrap to capture a clean OOM into the result (baseline) rather than crash.
3. Drive `runner.stream()` for the single configured prompt, recording every `TokenEvent`.
4. Stop sampler; compute peak VRAM, peak RSS, peak system mem, and **integrate** power → GPU energy.
5. Emit one `RunResult` JSON line via the JSONL `StructuredLogger` to `results/<exp_id>.jsonl`.

**Matrix orchestration lives outside the harness** (in `cli`/`scripts`), because a true *cold* run
needs an empty OS page cache that an in-process loop cannot provide — the first pass would warm the
cache and corrupt every later "cold" reading. So the orchestrator runs **each scenario in an isolated
subprocess**, and before a `cold` scenario it **explicitly flushes the OS page cache** (Windows:
`EmptyStandbyList`/RAMMap CLI — documented in the repro instructions), then runs the paired `warm`
scenario immediately after on the now-populated cache. **Cold** = flushed-cache fresh process;
**warm** = the populated-cache repeat (D4 page-cache contrast). Process isolation also cleanly bounds
the baseline OOM and each run's peak-memory accounting.

### `RunResult` data contract (the JSON schema everything downstream reads)
```jsonc
{
  "schema_version": "1.0",
  "exp_id": "...", "runner": "airllm", "quant": "nf4",
  "model": "Qwen2.5-32B-Instruct", "param_count": 32_000_000_000,
  "prompt_tokens": 256, "max_new_tokens": 8, "phase": "warm",
  "ok": true, "error": null,                       // baseline OOM → ok:false, error:"OutOfMemoryError"
  "ttft_s": 12.4,
  "itl_s": [9.1, 9.0, 9.3, ...],                   // full series, not just the mean
  "tpot_s": 9.12, "throughput_tok_s": 0.109,
  "peak_vram_mb": 4180, "peak_rss_mb": 21300, "peak_sys_used_mb": 30100, "peak_sys_cached_mb": 15800,
  "gpu_energy_j": 8123.0, "cpu_energy_j_est": 41000.0, "runtime_s": 75.2,
  "perplexity": 7.84,                              // null on baseline
  "env": {"os": "...", "cuda": "...", "driver": "...", "torch": "..."}
}
```
Metrics/economics/roofline consume **only** this; they never touch a model.

---

## 5. Analysis modules (Tier 1 — pure, keyless)

- **`metrics/`** — derive/aggregate from `RunResult`: mean/percentile ITL, throughput, TTFT-vs-length
  curve, cold/warm deltas, perplexity table. No recomputation of timings (those are raw).
- **`economics/`** — `costing.cost_api(tokens_in, tokens_out, rate)` (lifted from HW2 `cost_of`);
  `cost_onprem(throughput, energy, tariff, capex, lifespan, utilization)`; `breakeven(volume) ->`
  cumulative-cost curves for the five lines (API · API+cache · On-Prem realistic · On-Prem AirLLM ·
  Cloud GPU). All knobs from `economics.json` (D6, D8).
- **`roofline/`** — ceilings: compute (FP16 TFLOP/s), HBM (~912 GB/s), PCIe (~32 GB/s), NVMe
  (~5–7 GB/s). Operating points from measured throughput + `param_count`: FLOPs/token ≈ 2·P,
  bytes/token ≈ P·bytes_per_weight, achieved FLOP/s = FLOPs/token · throughput. Places baseline-GPU
  prefill/decode, llama.cpp, AirLLM-FP16/NF4-cold (NVMe), AirLLM-NF4-warm (PCIe/RAM) (D7). Arithmetic
  emitted alongside each point.
- **`plotting/`** — reads `results/` + analysis outputs → PNGs in `figures/`. Pure; `make figures`
  regenerates all offline.

---

## 6. Config schemas (versioned, fail-loud — D8)

Loader raises `ConfigVersionError` on a `"version"` prefix mismatch (ported pattern).
- **`config/setup.json`** — hardware spec (CPU/GPU/VRAM/RAM/NVMe), model registry, paths
  (`layer_shards_saving_path` → NVMe).
- **`config/economics.json`** — API price constants (with `source`/date comments), electricity
  tariff, CAPEX + scope, lifespan, utilization, cloud $/GPU-hr, caching discount. **No hardcoded
  numbers in code** (CI: `check_no_hardcoded.py`).
- **`config/experiments/<id>.json`** — per-scenario: runner, quant, prompt set, length sweep,
  `max_new_tokens`, repetitions. *Data, not code* — the matrix is sized post-spike.

---

## 7. CI & verification (D11)

| Job | Tier | Checks |
|-----|------|--------|
| lint/type | 1 | ruff zero · mypy --strict · file-size ≤150 |
| test | 1 | pytest ≥90% on deterministic core; **mock-runner** harness smoke; structural evals |
| structural evals | 1 | harness captures all fields · break-even monotonic · roofline points correct from inputs |
| `check_report.py` | 1 | README has every required section + figure embedded |
| `check_raw_data.py` | 1 | every figure has backing rows in `results/` |

CI **never** runs a real model (Tier 2). Mock runner proves wiring keylessly.

---

## 8. Testing strategy

- Pure modules (metrics/economics/roofline) → deterministic unit tests on fixture `RunResult` JSON.
- Harness → tested against the `mock` runner (no GPU).
- Runners → thin; their real exercise is the manual Tier-2 run, committed as evidence.
- Fixtures via the ported `conftest.py` config-file pattern.

---

## 9. Build order (feeds `TODO.md`)

Phase 1 (pre-spike): governance + `pyproject` + `uv sync`, CI green on empty src/tests.
Phase 2: **spike** (throwaway).
Phase 3 (post-spike): `shared/` + `costing` + `logging` + checkers + doc skeletons.
Phase 4: `Runner` Protocol + `mock` → `harness` + `ResourceSampler` → `metrics` (all keyless-testable).
Phase 5: real runners (`baseline_hf`, `airllm`, `llamacpp`) + the Tier-2 runs.
Phase 6: `economics` + `roofline` + `plotting`.
Phase 7: README report + `check_report`/`check_raw_data` green.

---

## 10. Definition of done (PLAN gate)

Approved when Imree signs off that the module boundaries, the `Runner`/`RunResult` contracts, the
harness/sampler design, the config schemas, and the CI tiers correctly realize the PRD. On approval
→ `TODO.md` gate (atomic task breakdown) on branch `docs/todo`.
