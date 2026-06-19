# PRD — AirLLM Benchmark (`airllm-bench`)

**Course:** Orchestration of AI Agents (203.3763) · Dr. Yoram Segal · HW5 / EX05
**Team:** Imree Cohen (driver) + Eyal Shtinmetz (reviewer) · **Method:** Vibe Coding, AI-agent grading
**Status:** Draft — awaiting PRD-gate approval. No production code until approved.
**Source decisions:** locked in the pre-build grill session (13 decisions, recorded below).

---

## 1. Mission

Take an LLM that is **too large** for our hardware, show that direct inference fails
(the baseline OOM), then use **AirLLM + quantization** to make the same model run — and
analyze the cost/benefit in depth, technically and economically. The graded deliverable is a
**deep-dive technical report (the README *is* the report)** backed by reproducible measurement
scripts and committed raw data.

The grader scores the **envelope** — measurement rigor, reproducibility, theory-linked analysis,
transparent economics — **not whether AirLLM "won."** A well-analyzed negative result is
explicitly welcomed by the brief and is, in fact, our expected headline finding.

**Report thesis (framing, not a hoped-for outcome):** *"When does layer-by-layer streaming
actually pay off?"* — answered with measured data, the roofline as the unifying framework, and
the economics as the consequence.

---

## 2. Hardware (the axis everything hangs on)

| Component | Spec | Role in the experiment |
|-----------|------|------------------------|
| GPU | RTX 3080 Ti, **12 GB VRAM** | The capacity wall the baseline must hit (OOM target). |
| RAM | **32 GB** | Model must exceed this so CPU-offload can't silently rescue the baseline. |
| CPU | Ryzen 9 5900X, 12 cores | CPU-fallback path; CPU power estimated by TDP. |
| Storage | **NVMe SSD** (932 GB) + slow HDD (1.82 TB) | AirLLM `layer_shards_saving_path` **MUST** point at the NVMe, never the HDD. |
| OS | **Windows 10** | Primary platform risk — see R1. WSL2 is a documented fallback only. |

---

## 3. The 13 locked decisions (requirements)

### D1 — Mandatory gated verification spike (FR-SPIKE)
A hard go/no-go spike runs **before any production code or model lock**. On a *tiny* model first,
it must prove: (a) the stack imports and runs on **native Windows** (torch+CUDA sees the GPU;
AirLLM imports; bitsandbytes `4bit`/`8bit` loads a layer); (b) the harness instrumentation works
end-to-end (streaming per-token timestamps, NVML VRAM/power, psutil RAM, JSON output);
(c) **one real token** is generated from Qwen2.5-32B via AirLLM and **timed**; (d) the OS
page-cache effect (does NF4 fit in 32 GB across passes?) is observed; (e) the AirLLM wrapper
returns `logits` for a full-sequence forward pass (needed for perplexity). **WSL2 fallback** is
pre-decided but dispreferred (it distorts the disk-I/O numbers that are the point).

### D2 — Model: Qwen2.5-32B-Instruct (FR-MODEL)
Target = **Qwen2.5-32B-Instruct** (FP16 ≈ 64 GB: 5× VRAM, 2× RAM — a genuine OOM no offload
can rescue). Fallback policy: **cut output tokens before cutting model size.** Drop to
Qwen2.5-14B-Instruct only as a last resort, documented as an explicit baseline-weakening tradeoff.
72B is rejected (brief warns against "no chance even with AirLLM").

### D3 — Scenario matrix includes the realistic competitor (FR-SCENARIOS)
| # | Scenario | Runtime | Purpose |
|---|----------|---------|---------|
| 1 | Baseline FP16 direct → OOM | HF `transformers` → GPU | The VRAM wall (spec-required; yields no tokens). |
| 2 | AirLLM FP16 | AirLLM | Worst latency; doesn't fit RAM → disk-streaming I/O bound. |
| 3 | AirLLM INT8 | AirLLM (bitsandbytes) | Less I/O. |
| 4 | AirLLM NF4 (4-bit) | AirLLM (bitsandbytes) | Least I/O; may fit RAM page cache → non-linear speedup. |
| 5 | **llama.cpp/Ollama Q4_K_M + GPU offload** | llama.cpp | **The realistic competitor** — core, not optional. |

Scenarios 2–4 are a **within-runtime quant sweep** (same algorithm, only bit-width changes);
scenario 5 is a **cross-runtime systems comparison** at "4-bit-class." These are labeled
distinctly — the quant schemes (bitsandbytes NF4/INT8 vs GGUF K-quants) are **not** apples-to-apples.

### D4 — Theory-revealing instrumentation (FR-METRICS)
The single harness every run passes through must:
- Derive **TTFT and TPOT from per-token streaming timestamps**, never total-time ÷ tokens.
- Run a **≥3-point input-length sweep** (~32 / ~256 / ~1024 tokens) to *empirically prove*
  prefill scales with input length while TPOT stays ~flat.
- Measure **cold and warm** deliberately (the gap is the paging/page-cache phenomenon).
- Use **greedy decoding** (temperature 0) for deterministic timing and output.
- Record the full **ITL series** (not just the mean) — AirLLM's is expected bimodal (cache hit/miss).

### D5 — Measured power + honest memory accounting (FR-RESOURCE)
- **GPU energy = measured** via NVML (`nvmlDeviceGetPowerUsage`) integrated over the run.
- **CPU energy = TDP-based estimate**, assumption declared.
- **Peak VRAM** via NVML (not `torch.cuda`, which misses bitsandbytes/AirLLM allocations).
- **RAM:** report **both** process RSS (psutil) and system used/cached (`virtual_memory`), with
  explicit definitions — the gap under mmap *is* the memory-hierarchy evidence.
- One background sampler thread, timestamped to the token-stream clock.

### D6 — Economics: two on-prem lines + honest quality caveat (FR-ECON)
- **Primary on-prem line** driven by the **realistic deployment throughput** (scenario 5).
- **Second cautionary on-prem line** = AirLLM, to show with numbers it never amortizes.
- API priced on the **closest-comparable model** (Qwen2.5-32B/72B), with an explicit
  **quality-parity caveat** (4-bit local ≠ frontier API); optional frontier reference point.
- Chart lines: API · API+caching · On-Prem (realistic) · On-Prem (AirLLM, cautionary) · Cloud GPU.
- CAPEX scope (whole-PC vs GPU), lifespan, **utilization** (the most sensitive knob), and
  electricity tariff are **all exposed config knobs** in `economics.json`.
- On-prem electricity cost is driven by **measured** energy (D5).

### D7 — Hierarchical roofline as the centerpiece (FR-ROOFLINE)
Build the **textbook GPU roofline first** (compute ceiling + HBM ~912 GB/s diagonal) to validate
the llama.cpp/GPU prefill (compute-bound) and decode (memory-bound) points against theory. Then
extend to a **hierarchical multi-ceiling roofline** (compute → HBM → PCIe → NVMe) and place
**AirLLM decode on the NVMe diagonal** (~200× below the silicon roof). Operating points computed
from measured throughput + known param count, **with the arithmetic shown**. This chart is the
report's unifying figure.

### D8 — No live pricing; Gatekeeper N/A (FR-CONFIG)
No live API calls of any kind. **API prices are documented constants in `economics.json`** (each
with source + retrieval date); token counts come from the **local tokenizer**. Gatekeeper recorded
as **N/A in an ADR** (no live third-party calls; local inference lives openly in `runners/`).
`.env` needs only `HF_TOKEN`. **Truly keyless by default.**

### D9 — Right-sized architecture; QLoRA deferred (FR-ARCH)
**Keep:** thin runner `Protocol` (`run(prompt, config) -> RunResult`, three implementations);
config-as-data (versioned, fail-loud); JSON/JSONL results with plotting derived *from* results;
`check_report.py` + `check_no_hardcoded.py`; CI lint/type/test + file-size guard.
**Cut:** the SDK-facade ceremony and provider-abstraction layers (no providers exist here). A small
CLI with subcommands (`bench`, `figures`, `economics`, `report-check`) over plain functions.
**QLoRA second extension = out of scope**, documented in `EXTENDING.md` as future work.

### D10 — Perplexity as primary quality metric (FR-QUALITY)
**Perplexity** on a fixed held-out passage (one cheap forward pass on AirLLM, *not* autoregressive),
per quant level, **both runtimes** — gives a cross-runtime quality axis and a number for the D6
quality-parity caveat. **Plus** a small sample-output table for the brief's literal "qualitative"
wording. (Logits availability verified in D1.)

### D11 — Two-tier reproducibility (FR-REPRO)
- **Tier 1 (keyless, offline, CI):** raw JSON = source of truth; **all figures regenerate from
  committed JSON** (`make figures`); structural evals + `check_report.py` + `check_raw_data.py`
  (every figure has backing data) + a **mock runner** smoke test for harness wiring.
- **Tier 2 (hardware-bound):** the model→data step is documented (commands, pinned env, driver/CUDA
  versions) and committed as evidence under `results/`; **never described as CI-reproducible.**
- No inherited "clean-room CI rebuilds the artifact" language (it would be a doc↔repo contradiction).

### D12 — Negative result presented as characterization, not failure (FR-NARRATIVE)
Public docs frame AirLLM's likely loss as a **measured characterization of its operating envelope**
("runs-at-all vs doesn't-run for 70B+; dominated by offload for 32B — here's the roofline reason"),
never as a project failure. No confessional/self-undercutting framing in graded docs; candid
strategy lives in untracked `*.local.md` notes.

### D13 — Honest ownership + calibrated self-grade (FR-PROCESS)
Ownership represented **consistent with `git shortlog`**: Imree drives + authors most commits; Eyal
reviews (Claude may *draft* review content, but Eyal genuinely engages and signs off under his own
name — no fabricated authorship). `SELF_GRADE.md` (per-requirement) + `KNOWN_LIMITATIONS.md`
(severity + fix sketch). **Report 92–93**, cap 95.

---

## 4. Sequencing (carry-over is split around the spike)

1. **Phase 1 — minimal bootstrap (before spike):** governance shell (CLAUDE.md, Makefile, CI,
   pre-commit, `.gitignore` + model-weight lines, `.env.example`), `pyproject.toml` (heavy deps),
   `uv sync`. CI green on empty `src`/`tests`.
2. **Phase 2 — THE SPIKE (go/no-go gate D1).** Throwaway scripts only.
3. **Phase 3 — full carry-over (after spike green):** config loader + models, `logging_setup`,
   `costing.py` (lifted `cost_of`), `conftest`, `check_no_hardcoded`, `check_report.py`,
   `check_raw_data.py`, doc skeletons.
4. **Phase 4+ — new build:** harness → runners → metrics → economics → roofline → report.

Rationale: porting the full scaffold before the spike would polish an unproven foundation. Thin
slice (env + governance) gates the spike; the bulk lands once viability is proven.

---

## 5. Deliverables (from the brief)

- [ ] Full experiment code + reproducible run/measure scripts.
- [ ] Deep-dive technical report = README (hardware, baseline, AirLLM, quantization).
- [ ] Comparative tables + graphs of performance metrics, **embedded inline** in the README.
- [ ] Economic analysis (API vs On-Prem + Cloud + caching), break-even chart, all assumptions stated.
- [ ] Theory-linking analysis (Prefill/Decode, memory/compute-bound, VRAM, paging/mmap).
- [ ] Original extension documented (the hierarchical roofline).
- [ ] Reproduction instructions + hardware spec + model-choice justification.

---

## 6. Scope

**In:** scenarios 1–5; FP16/INT8/NF4 AirLLM sweep + Q4_K_M competitor; the six metrics + perplexity;
two-line economics + caching + cloud; hierarchical roofline; two-tier reproducibility.

**Out:** QLoRA fine-tune (deferred); live API pricing/inference; multi-model size comparison as the
primary extension; running the benchmark in CI; any model weights committed to git.

---

## 7. Open items (deliberately unresolved — gated on the spike)

- Final model lock (32B vs 14B fallback) — decided by **measured** spike token-time.
- Scenario matrix size: `max_new_tokens`, prompt count, repetitions — sized to measured decode time.
- Native Windows vs WSL2 — decided by spike install/run result.

Committing these now would be the napkin-math mistake the brief warns against.

---

## 8. Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R1 | Windows + bitsandbytes/AirLLM won't install/run | 🔴 | Spike gates it; WSL2 documented fallback. |
| R2 | Decode latency makes even a tiny matrix span hours | 🔴 | Spike measures token-time; matrix sized to it; tiny token budgets. |
| R3 | Roofline placed on the wrong ceiling | 🟡 | Build textbook-first then extend; show the arithmetic. |
| R4 | mmap RAM accounting misread | 🟡 | Report both RSS and system/cached with definitions. |
| R5 | AirLLM wrapper doesn't expose logits (breaks perplexity) | 🟡 | Verified in spike (D1e). |
| R6 | Scaffolding/process layer | 🟢 | Solved by salvage from HW2/HW3. |

---

## 9. Definition of done (for the PRD gate)

This PRD is approved when Imree signs off that the 13 decisions, the spike-gated sequencing, and the
scope boundaries above correctly capture the agreed plan. On approval → `PLAN.md` gate (architecture,
contracts, C4) on branch `docs/plan`.
