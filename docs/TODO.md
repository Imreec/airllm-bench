# TODO — Atomic task tracker (`airllm-bench`)

**Status:** Draft — awaiting TODO-gate approval. Derives from `PRD.md` (D1–D13) + `PLAN.md`.
**Rule:** each task is one atomic PR (single concern), TDD where code, green gate every push.
Two hard gates: **G-SPIKE** (go/no-go) and the per-PR review gate.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · **(ref)** = PRD/PLAN anchor.

---

## Phase 1 — Pre-spike bootstrap (enables the spike)

- [ ] **T1.1** Port governance shell: `CLAUDE.md`, `Makefile`, `.pre-commit-config.yaml`,
      `.gitattributes`. **(PLAN §2)**
- [ ] **T1.2** `.gitignore` + **model-weight lines** (`*.safetensors`, shard dirs, HF cache,
      large `results/*`); `.env.example` = `HF_TOKEN` only. **(D8)**
- [ ] **T1.3** `pyproject.toml`: keep ruff/mypy/pytest/coverage config; rewrite deps
      (torch, transformers, accelerate, airllm, bitsandbytes, llama-cpp/ollama client, matplotlib,
      pandas, psutil, pynvml). `uv sync`. **(salvage)**
- [ ] **T1.4** CI `quality.yml`: lint + mypy --strict + pytest + file-size ≤150. Green on **empty**
      `src/airllm_bench/` + `tests/`. **(D11, PLAN §7)**

**→ Phase 1 merged before any spike work.**

---

## Phase 2 — G-SPIKE (go/no-go, throwaway scripts only) **(D1)**

- [ ] **T2.1** Stack import check on **native Windows**: torch+CUDA sees the 3080 Ti; AirLLM imports;
      bitsandbytes `4bit`/`8bit` loads a layer. (WSL2 fallback decision if it fails.)
- [ ] **T2.2** Tiny-model harness proof (Qwen2.5-0.5B/7B): streaming per-token timestamps,
      NVML VRAM/power, psutil RSS+system, JSON out — full instrumentation works end-to-end.
- [ ] **T2.3** **One real Qwen2.5-32B token via AirLLM, timed** → replaces the decode-time estimate.
- [ ] **T2.4** Page-cache check: does NF4 (~16 GB) stay resident across passes in 32 GB?
- [ ] **T2.5** Verify AirLLM exposes `logits` for a full-sequence forward pass (perplexity). **(D10)**

**🚦 G-SPIKE decision (recorded in an ADR):** lock model (32B vs 14B), size the matrix
(`max_new_tokens`, prompts, reps), confirm native-Windows vs WSL2. **(PRD §7)** No further code
until green.

---

## Phase 3 — Post-spike carry-over

- [ ] **T3.1** `shared/config.py` + `config_models.py` — versioned, fail-loud (`ConfigVersionError`),
      trimmed to HW5 schema. **(D8, PLAN §6)**
- [ ] **T3.2** `shared/logging_setup.py` — JSONL `StructuredLogger` (lifted, secret-redaction kept).
- [ ] **T3.3** `economics/costing.py` — lift `cost_of` from HW2 `budget.py`, drop enforcement. **(D6)**
- [ ] **T3.4** `config/setup.json` + `config/economics.json` (priced constants w/ source+date) +
      one `config/experiments/<id>.json` template. **(D6, D8)**
- [ ] **T3.5** `conftest.py` fixture; `scripts/check_no_hardcoded.py` (watched constant repointed),
      `check_file_sizes.py`. **(PLAN §7)**
- [ ] **T3.6** Doc skeletons (content wiped): `SELF_GRADE.md`, `KNOWN_LIMITATIONS.md`, `COST.md`,
      `EXTENDING.md` (note QLoRA as future work), `PROMPTS.md` (fresh), `adr/` (Gatekeeper-N/A ADR,
      G-SPIKE ADR). **(D8, D9, D13)**

---

## Phase 4 — Harness core (keyless-testable, no GPU needed)

- [ ] **T4.1** `runners/protocol.py` — `Runner` Protocol, `TokenEvent`, `LogitsResult`,
      `RunResult` dataclass + JSON (de)serialize. **(PLAN §3, §4)**
- [ ] **T4.2** `runners/mock.py` — canned `TokenEvent`s for wiring tests.
- [ ] **T4.3** `harness/sampler.py` — `ResourceSampler` thread (NVML power/VRAM, psutil RSS+system),
      unit-tested with a fake clock. **(D5)**
- [ ] **T4.4** `harness/run.py` — single-scenario `run(runner, cfg) -> RunResult`; clean-OOM capture;
      tested against `mock`. **(PLAN §4)**
- [ ] **T4.5** `metrics/` — TTFT/TPOT/throughput, ITL series stats, TTFT-vs-length curve, cold/warm
      delta, perplexity from `LogitsResult`. Pure, unit-tested on fixtures. **(D4, D10)**
- [ ] **T4.6** Structural evals + `scripts/check_raw_data.py` + mock-runner CI smoke. **(D11)**

---

## Phase 5 — Real runners + Tier-2 measured runs (hardware-bound)

- [ ] **T5.1** `runners/baseline_hf.py` — FP16, **GPU-only** (`device_map={"":0}`), expected clean OOM
      at load. **(D3, PR-review fix)**
- [ ] **T5.2** `runners/airllm.py` — AutoModel path, `compression` = none/8bit/4bit, shards → **NVMe**.
- [ ] **T5.3** `runners/llamacpp.py` — GGUF Q4_K_M (+Q8), `n_gpu_layers` offload. **(D3)**
- [ ] **T5.4** `cli`/`scripts` matrix orchestrator: per-scenario **subprocess isolation** +
      **OS page-cache flush** before cold. **(PLAN §4, PR-review fix)**
- [ ] **T5.5** Execute the full matrix; commit raw JSON to `results/` as evidence + env metadata.
      **(D11 Tier-2)**

---

## Phase 6 — Analysis & figures (keyless, from committed JSON)

- [ ] **T6.1** `economics/` — on-prem (CAPEX+OPEX, measured energy), two on-prem lines, API+caching,
      cloud; break-even curves. **(D6)**
- [ ] **T6.2** `roofline/` — compute/HBM/PCIe/NVMe ceilings; operating-point placement (FP16+NF4-cold
      → NVMe, NF4-warm → PCIe/RAM); arithmetic emitted. **(D7, PR-review fix)**
- [ ] **T6.3** `plotting/` — all figures from `results/`: TTFT-vs-length, ITL, throughput, perplexity,
      break-even, hierarchical roofline. `make figures` regenerates offline. **(D11)**

---

## Phase 7 — Report & final gates

- [ ] **T7.1** `scripts/check_report.py` — assert README has every required section + embedded figure.
      **(salvage, D11)**
- [ ] **T7.2** Write `README.md` = the deep-dive report: hardware + model justification, experiment,
      findings (baseline/AirLLM/quant/competitor), theory-linking (Prefill/Decode, memory/compute-
      bound, paging/mmap), economics + recommendation, reproduction, all figures inline. Framed as
      "where does layer-streaming pay off?" **(D12, brief §8)**
- [ ] **T7.3** Fill `KNOWN_LIMITATIONS.md` + `SELF_GRADE.md` (target 92–93) + `COST.md`. **(D13)**
- [ ] **T7.4** Final integration gate: `make grade` + `check_report` + `check_raw_data` green;
      every figure backed by raw data; docs↔repo consistent. **(D11, D13)**

---

## Definition of done (TODO gate)

Approved when Imree signs off that the phases, atomic tasks, ordering, and the two hard gates
(G-SPIKE + per-PR) correctly sequence the build. On approval → Phase 1 begins on a feature branch.
