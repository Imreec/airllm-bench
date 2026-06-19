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
