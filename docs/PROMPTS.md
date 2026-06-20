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
