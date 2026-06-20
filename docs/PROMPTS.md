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
