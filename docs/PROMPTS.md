# Prompt Log

Truthful, continuous record of the prompts that drove **committed** work (CLAUDE.md §8). Driver:
Imree (Claude Code session). Reviewer: Eyal + a separate-model PR reviewer. Only shipped work is
logged; entries are added in the PR that does the work.

---

### Pre-planning — grill / pressure-test
Drove a one-decision-at-a-time interrogation of the suggested plan before any code (13 decisions
locked). Outcome: the PRD's D1–D13, incl. adding the llama.cpp competitor, the hierarchical roofline,
dropping live pricing, and the spike-first gate.

### PR #1 — PRD
"Turn the grill outcome into a PRD." → `docs/PRD.md` (13 decisions, spike as go/no-go, deliverables).
Cross-model review caught the baseline-OOM `device_map` hazard and refined the NF4 roofline ceiling.

### PR #2 — PLAN
"Translate the PRD into architecture + contracts." → `docs/PLAN.md` (C4, `Runner`/`RunResult`).
Review (blocking) caught that an in-process length×phase sweep corrupts cold runs → one-scenario-per-
`RunResult` + external orchestration; plus native per-runtime tokenization for perplexity.

### PR #3 — TODO
"Sequence PRD+PLAN into atomic tasks." → `docs/TODO.md` (7 phases). Review (blocking) caught the WSL2
RAM-cap and the admin requirement for the cold-cache flush.

### PR #4 — Phase 1 bootstrap
"Port the governance shell + deps + CI." → CLAUDE.md, Makefile, pre-commit, CI, pyproject (core/runtime
split), adapted `.claude/skills/`. Corrected HW2-isms in the workflow docs (squash-merge, real review
history).

### PR #5 — G-SPIKE
"Write throwaway go/no-go scripts; run them on the box." → `spike/`. Surfaced and fixed a cascade
(CPU torch → CUDA index; transformers 5.x/4.43 → pin `<4.43`; optimum 2.0; sentencepiece; shards-dir;
logits tuple). Result: GO, native Windows, 32B @ 4-bit ≈ 20 s/token (ADR 0001).

### PR #6 — Phase 3 foundation
"Port config loader, models, JSONL logger, costing." → `shared/` + `economics/costing.py` + `config/`.
14 tests, 100% coverage.

### PR #7 — Phase 3 checkers + docs
"Port `check_no_hardcoded` + `conftest`; seed doc skeletons + Gatekeeper-N/A ADR." → this PR.

### PR #8 — Phase 4 part 1: runner contract
"Build the runner contract + wiring fixtures (keyless)." → `runners/protocol.py` (`Runner` Protocol,
`TokenEvent`, `LogitsResult`), `harness/result.py` (`RunResult`), `runners/mock.py`, `harness/sampler.py`
(`ResourceSampler`, fake-clock tested). No GPU, no key.

### PR #9 — Phase 4 part 2: harness + metrics
"Wire the one-scenario harness + the pure analysis metrics + structural evals." → `harness/run.py`
(one scenario → one `RunResult`, clean-OOM capture), `metrics/` (timing/aggregate/quality, perplexity),
structural evals + `check_raw_data.py` + the mock-runner CI smoke test.

### PR #10 — Phase 5 (T5.1): baseline_hf runner
"Build the three real runners as thin adapters, unit-tested keyless by mocking torch/airllm." → first
runner: `runners/baseline_hf.py` (FP16 GPU-only `device_map={"":0}`, expected clean VRAM OOM) + the
shared `runners/hf_decode.py` (greedy decode + forward-logits, DRY for the AirLLM runner). Tests inject
fake torch/transformers via `sys.modules` so CI stays keyless. Also backfilled the PR #8/#9 prompt log.

### PR #11 — Phase 5 (T5.2): airllm runner
"Build the AirLLM runner reusing the shared decode core." → `runners/airllm.py` (AutoModel path,
`cfg.quant` → `compression` none/8bit/4bit, pre-creates the shards dir for AirLLM's `check_space`,
logits from the forward tuple `out[0]`). Reuses `hf_decode`; tests inject a fake `airllm` module and
capture `from_pretrained` kwargs to assert the compression mapping. Keyless, 100% coverage.

### PR #12 — Phase 5 (T5.3): llamacpp runner
"Build the llama.cpp GGUF competitor." → `runners/llamacpp.py` (token-by-token `Llama.generate` at
temp=0, EOS/budget stop; `logits_all` eval for perplexity; partial GPU offload via `n_gpu_layers`).
Using the low-level `generate` API gives real token ids, so the earlier streaming-token-id caveat
dissolved — no placeholder needed. Tests inject a fake `llama_cpp`; keyless, 100% coverage. Completes
the three real runners (T5.1–T5.3); next is the T5.4 matrix orchestrator.

### PR #13 — Phase 5 (T5.4a): scenario worker + runner factory
"Execute one scenario from config, keyless." → `runners/factory.py` (build the right runner with
device/shards/gguf/n_gpu_layers injected from `setup.json`) + `harness/scenario.py` (`run_scenario`:
build → run → append one RunResult JSONL line; sampler/env injected so it stays keyless). Config grows
a `runners` section + an optional `prompt_tokens` override (whitespace estimate when absent). Mock-runner
tested, 100% coverage. T5.4b adds the hardware glue (NVML reader, env, cold-cache flush + admin guard,
the subprocess matrix orchestrator, `cli bench`).

### PR #14 — Phase 5 (T5.4b): hardware instrumentation
"Capture GPU+memory telemetry and stack metadata into each RunResult." → `harness/hw_reader.py`
(`make_reader`: NVML power/VRAM + psutil RSS/system → the `ResourceReading` the sampler polls; lazy
import, fakes injected in tests) + `harness/env_info.py` (`collect_env`: os/python/torch/cuda/driver,
best-effort so missing pieces are "" not errors). Caught an env-dependent test (the dev box has a real
GPU, so NVML succeeds locally) → forced the absent path with `sys.modules[name]=None`. 100% coverage.
T5.4c adds the cold-cache flush + admin guard + the subprocess matrix orchestrator + `cli bench`.

### PR #15 — Phase 5 (T5.4c): matrix orchestrator + cold-cache flush + cli
"Drive the cold/warm matrix with subprocess isolation." → `harness/cache_flush.py` (fail-loud
`require_admin` for the Standby-List flush, `flush_standby_list`, `verify_cache_dropped`,
`build_cold_flush`), `harness/orchestrator.py` (`run_matrix`: flush before each cold scenario, spawn
each in a fresh subprocess, `tqdm`), `harness/scenario_main.py` (subprocess entrypoint wiring the NVML
sampler + env into the keyless worker), and a thin `cli.py` (`bench`). `tqdm` declared in CORE (the
keyless orchestrator smoke test imports it). OS-specific bits all injected → 100% coverage on every new
module; only the Windows admin-check + `__main__` shims are pragma-excluded. Completes T5.4 — Phase 5
is now code-complete; T5.5 is the at-the-keyboard Tier-2 run.

### PR #16 — Phase 5 (T5.5 prep): experiment matrix + runbook
"Size the Tier-2 matrix and write the configs + runbook before the at-the-keyboard run." → 13
single-scenario `config/experiments/*.json` (AirLLM 4-bit cold/warm + warm rep + TTFT-vs-length sweep
at 64/256/1024 native tokens, 8-bit cold/warm, FP16 cold/warm, llama.cpp warm + rep, baseline OOM),
prompt_tokens pinned by tokenizing with the real Qwen tokenizer. `docs/RUNBOOK_TIER2.md` captures the
disk choreography (C: NVMe for shards+GGUF, D: HDD for the HF cache only; one shard set at a time;
pre-shard before cold so sharding doesn't warm the cache; elevated shell for the flush) + the batch
commands. Adds the `airllm-bench` console entrypoint. Keyless; 91 tests green.

### PR #17 — Phase 5 (T5.5 enablement): llama.cpp CUDA DLL shim
"Make the CUDA llama.cpp wheel load on the box." → during T5.5 prep the PyPI `llama-cpp-python` proved
CPU-only, so swapped in the cu124 prebuilt wheel (0.3.4); its `ggml-cuda.dll` couldn't find the CUDA
runtime DLLs. `runners/llamacpp.py` now adds torch's bundled `torch/lib` (which ships cudart/cublas
cu124) to the DLL search path before importing `llama_cpp` — guarded `if sys.platform == "win32"` so
Linux mypy/CI stays clean. Verified on hardware: GPU offload True, 4 real tokens generated, logits
5×152064 for perplexity. Keyless; mypy clean on `--platform linux` and `win32`; 92 tests.

### PR #18 — Phase 5 (T5.5): capture resource peaks on the OOM path
"The baseline OOM came back with peak_vram_mb=null." → `harness/run.py` now attaches the sampler's
resource peaks (peak VRAM / RSS / system / energy) on the failure path too, not just on success — a
clean OOM is itself a resource event, and peak VRAM at the wall is the capacity-wall evidence (D3).
Found while running the real baseline scenario during T5.5 (peak_vram≈12.16 GB, the card maxed).
Keyless test on the OOM+sampler path; 92 tests, mypy clean (linux + win32).

### PR #19 — Phase 5 (T5.5): fix cold-flush verification for Windows reality
"The pre-flight cold-flush failed: freed only 30 MB (< 2000 required)." → the `verify_cache_dropped`
gate was wrong on Windows — the Standby List counts toward `available`, so emptying GBs barely moves
`available`, and the check could never pass. Removed the freed-MB gate (and `verify_cache_dropped` /
the available read / `min_rise_mb`); `build_cold_flush` now = require-admin → RAMMap `-Et` (exit 0 still
fails loud). Cold-ness is evidenced by the cold/warm timing delta (ADR 0001). Repointed config to
`RAMMap64.exe -accepteula -Et`; documented as L-07. Caught by the operator's pre-flight test — 90 tests.

### PR #20 — Phase 5 (T5.5): capture generation-time errors + drop len1024
"The 1024-token sweep crashed: tensor a (1024) must match tensor b (512)." → AirLLM's default
`max_seq_len=512` rejects a 1024-token prompt, and the harness only wrapped *load* errors, so the
crash killed the subprocess and lost the scenario. `harness/run.py` now also captures *generation*-time
failures into `ok=False` (protecting the ~70-min FP16 runs); `mock.py` gains a `stream_error` to test it.
Dropped the `len1024` config (the 40/64/256 points already show AirLLM's TTFT is streaming-bound);
documented as L-08. 92 tests.

### PR #21 — Phase 5 (T5.5): the Tier-2 measured results
"Run the matrix on the box and commit the evidence." → 12 `results/*.jsonl` RunResult rows measured on
the RTX 3080 Ti: AirLLM 4-bit/8-bit/FP16 cold+warm (the headline cold/warm contrast, run with the
**paired protocol** after the split approach was found to contaminate warm via cross-model page-cache),
the 4-bit length sweep, the llama.cpp Q4 competitor, and the baseline FP16 OOM. The story is clean and
monotonic: warm speedup 2.26× (4-bit, fits RAM) → 1.07× (8-bit) → 1.02× (FP16, exceeds RAM); TPOT and
perplexity both ordered by precision. `results/README.md` documents the matrix + env. Closes T5.5 —
Phase 5 complete; Phase 6 (economics/roofline/plotting) consumes these keylessly.

### PR #22 — Phase 6 (T6.1): economics
"Build the keyless economics layer from the committed results." → `metrics/load.py` (the one read path
for `results/*.jsonl`), `economics/config.py` (typed `EconomicsConfig` with provenance + CAPEX-scope
knob), `economics/energy.py` (per-token energy = measured GPU avg-power×tpot + TDP-estimated CPU),
`economics/onprem.py` (amortized CAPEX + measured-energy OPEX → $/Mtok; utilization the sensitive knob),
`economics/breakeven.py` (the five lines: API · API+cache · On-Prem realistic · On-Prem AirLLM · Cloud).
Pricing constants finalized with real sources (OpenRouter Qwen2.5-Coder-32B, Israel tariff, RunPod 3090)
— `economics.json` → 1.01; comparability caveats as L-09. **Finding:** at Israel's $0.228/kWh the local
single-stream 32B *electricity alone* (~$4.3/Mtok) exceeds the blended API price (~$1.66/Mtok), so
on-prem never breaks even — the API's batching/throughput advantage dominates (a clean negative result).
Structural eval builds the lines from committed config+results and asserts monotonic curves + the
cautionary AirLLM line dominating the realistic one. 116 tests, 100% on new modules, mypy clean.
Antigravity review caught a real [Blocking] bug — `…/(throughput or inf)` made a stalled run read as
$0 (free) not infinite; fixed + zero-utilization inf guard + breakeven_volume None on non-positive
crossing (PR #22 follow-up). 120 tests.

### PR #23 — Phase 6 (T6.2): roofline
"Build the hierarchical roofline from the committed results, keyless." → `roofline/ceilings.py`
(compute roof + HBM/PCIe/NVMe diagonals; `attainable_flops = min(compute, bw×intensity)`) and
`roofline/points.py` (operating point per run: FLOPs/tok=2·P, bytes/tok=P·bpw, intensity=2/bpw,
achieved FLOP/s & bandwidth; binding tier by physical data path — resident→HBM, AirLLM cold→NVMe,
warm→PCIe iff the footprint fits RAM, else NVMe — computed from `param_count·bpw < ram`). Ceilings
finalized with real sources in a `setup.json` `roofline` block (→1.01): 3080 Ti 68.2 TFLOP/s dense
tensor + 912 GB/s HBM, PCIe4 31.5 GB/s, NVMe ~7 GB/s, bytes/weight fp16=2/int8=1/nf4=0.5/q4_k_m≈0.56.
The nf4-warm point *climbs* HBM→PCIe exactly because 16 GB fits the 32 GB cache while int8/fp16 don't —
the memory-hierarchy result falls out of the arithmetic. Structural eval reproduces points from
committed config+results and asserts no point exceeds its binding bandwidth (utilization ≤ 1). Tier
assignment is by data path not saturation (AirLLM is overhead-bound ≪ every ceiling) — disclosed as
L-10. Antigravity review caught two [Blocking] gaps: (1) D7 requires a **prefill** (compute-bound)
point too, not just decode → added `prefill_operating_point` (FLOPs=2·P·prompt_tokens, intensity
2·tokens/bpw, achieved=FLOPs/ttft), with a structural assertion that prefill is higher-intensity than
decode; (2) `fits_in_ram` ignored OS/Python/torch RAM → added a configurable `os_overhead_gb` (≈5 GB,
usable cache ≈27 GB). Nit: moved the resident-runner knowledge out of code into `setup.json`
(`"resident": true`) read via `resident_runner_names`. 137 tests, 100% on new modules.

### PR #24 — Phase 6 (T6.3): plotting
"Render every report figure from committed data, keyless and offline." → `plotting/` (Agg-backed
`base`, `latency` (ttft-vs-length / ITL / cold-warm headline), `throughput_quality`, `economics_fig`
(break-even), `roofline_fig`, `prepare` (pure result→figure selectors), `build` (orchestrator)).
`make figures` + `airllm-bench figures` regenerate all 7 PNGs into `figures/` (committed). Figure
builders are pure functions of prepared data so each is unit-tested for the right artists; `build`
is exercised end-to-end on the committed results. `check_raw_data.py` tightened from "any figure
needs any data" to: the committed figure set must equal the expected set AND the headline scenarios
(llamacpp-q4-warm, airllm-nf4-cold/-warm) must exist under `results/`. The break-even chart shows the
on-prem line never dropping below the API line (never amortizes); the roofline shows decode points at
low intensity and prefill points climbing toward the compute roof. Antigravity review fixed three:
[Blocking] the hardcoded API model key → `economics.json` gains `active_model` (→1.02), read by
`build`; [Should-fix] roofline annotations collided (decode/prefill share exp_id) → annotate with
`(phase)`; [Should-fix] hardcoded `prompt_tokens=40` + asymmetric cold/warm queries → a derived
`_baseline_length` (min prompt length) applied symmetrically to both phases. 163 tests, mypy --strict
clean (linux + win32). Completes Phase 6 — only the README report + final gates (Phase 7) remain.
