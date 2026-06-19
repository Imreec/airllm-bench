# Project Conventions — `airllm-bench` (HW5)

> **Course:** Orchestration of AI Agents (203.3763), University of Haifa · **Instructor:** Dr. Yoram Segal
> **Source of truth:** Dr. Segal's "Guidelines for Writing Professional Software at the Highest Level
> of Excellence" (v3.00), tightened for this assignment.
> **Project character:** a **benchmarking & analysis** project, not a software-architecture one. The
> graded deliverable is a deep-dive technical report (the README) backed by reproducible measurement
> scripts and committed raw data. Decisions are locked in `docs/PRD.md` (D1–D13) + `docs/PLAN.md`.

Sections marked **[TIGHTENED]** raise the threshold above the course minimum; **[HW5]** are
assignment-specific.

---

## 1. Hard constraints — non-negotiable

| # | Rule | Threshold |
|---|------|-----------|
| 1 | Python source file size | ≤ 150 source lines (incl. tests) — **split, never compress** |
| 2 | Linter | `uv run ruff check .` → 0 violations |
| 3 | Test coverage (deterministic core) | ≥ 90% [TIGHTENED] |
| 4 | Package manager | `uv` only — `pip`/`venv`/`python -m` forbidden |
| 5 | Hardcoded values in code | 0 — config from JSON/env only |
| 6 | Secrets in code/repo | 0 — `.env` git-ignored, `.env.example` committed |
| 7 | Type hints on public APIs | 100% [TIGHTENED] |
| 8 | `mypy --strict` on `src/` | 0 errors [TIGHTENED] |
| 9 | Starting version | `1.00` (code + each config file) |
| 10 | Public function/method without a test | 0 |
| 11 | Code duplication (same logic in 2+ files) | 0 — extract to a module |
| 12 | CI green before any merge | required [TIGHTENED] |
| 13 | Model weights committed to git | 0 — tens of GB, `.gitignore`d |

---

## 2. Mandatory workflow — gates

**No production code until the planning docs are approved.** Status: PRD ✅, PLAN ✅, TODO ✅ (merged).

```
docs/PRD.md  → docs/PLAN.md  → docs/TODO.md   (each: explicit human "approved")
──────────────────────── ↓ only now ────────────────────────
Build per TODO phases (TDD red→green→refactor), the G-SPIKE gate first.
```

- The agent never proceeds past a gate without an unambiguous "approved" from Imree.
- The agent never claims work that isn't in the working tree; it runs the verification command and
  shows the output before reporting "done."
- TODO checkboxes are ticked as work lands, never as aspirations.

---

## 3. Architecture rules (as locked in PLAN)

- **No SDK-facade ceremony** (D9). Business logic lives in plain module functions; `cli.py` calls
  them directly. The one real abstraction is the **`Runner` Protocol** (three implementations + a
  mock) so every runtime is measured identically.
- **Measurement / analysis / presentation are separated.** Raw JSON under `results/` is the source of
  truth; `metrics`/`economics`/`roofline`/`plotting` are pure functions of committed data.
- **Two tiers** (D11): Tier-1 (analysis) is keyless, offline, CI-tested; Tier-2 (model runs) is
  hardware-bound and committed as evidence — never run in CI.
- **No `NotImplementedError` placeholders on `main`.** A function exists with a working body or not
  at all.
- **API Gatekeeper is N/A** (D8): no live third-party API calls. Economics uses documented price
  constants; local inference lives openly in `runners/`. Recorded in an ADR so drift checks don't
  false-flag.
- **DRY:** same body in 2+ files → shared module; same pattern in 3+ → wrapper/base/mixin.

---

## 4. Configuration & security

| Value type | Location |
|------------|----------|
| Model ids, paths, hardware spec | `config/setup.json` |
| Prices, tariffs, CAPEX, utilization, TDP | `config/economics.json` (each with source + date) |
| Per-scenario params (runner, quant, prompt, tokens) | `config/experiments/<id>.json` |
| Secrets (`HF_TOKEN`) | `os.environ` only |

- Every config JSON carries `"version"` starting `"1.00"`; the loader **fails loud**
  (`ConfigVersionError`) on mismatch.
- `.env` always git-ignored; `.env.example` committed (HF_TOKEN only — **keyless by default**: the
  whole Tier-1 suite + figures regenerate with no key).
- No hardcoded model strings, prices, or rates — `scripts/check_no_hardcoded.py` enforces it.

---

## 5. Testing — TDD + two validation layers

- Strict RED → GREEN → REFACTOR; tests before/alongside code, never after.
- Every public function has a test; happy path **and** error paths; external deps mocked.
- Test files also obey the 150-line limit.
- **Layer 1 (tests)** prove code runs. **Layer 2 (structural evals)** prove it behaves — deterministic,
  keyless invariants in CI (harness captures all fields, break-even monotonic, roofline points correct
  from inputs, every figure has backing raw data). The **Tier-2 measured runs** are committed as
  evidence under `results/`.
- A commit must never leave `main` in a broken-collection state.

---

## 6. Code quality

- Comments explain **why**, not what. Docstrings on every public function/class/module.
- Descriptive names; short single-purpose functions; consistent style.
- Type hints on every public API; `mypy --strict` clean on `src/`.
- `ruff` config in `pyproject.toml` (families: E,F,W,I,N,UP,B,C4,SIM,ANN,RET,PT).

---

## 7. Package management — `uv`

- `uv add` / `uv sync` / `uv run` — never `pip`/`python` directly. `uv.lock` committed.
- **Heavy model-runtime deps (torch, transformers, accelerate, airllm, bitsandbytes, llama-cpp) live
  in the `runtime` optional extra** — installed only on the hardware box (`uv sync --extra runtime`),
  **never in CI**. CI installs core + dev only, keeping Tier-1 fast and keyless.
- `requires-python`, ruff `target-version`, and mypy `python_version` kept in sync
  (`check_anti_patterns.py` enforces it). The exact version is confirmed by the G-SPIKE.

---

## 8. Documentation & disclosure

- `docs/PRD.md` / `PLAN.md` / `TODO.md` current; ADRs in `docs/adr/` (Context, Decision, Status,
  Consequences, Alternatives). The G-SPIKE outcome and the Gatekeeper-N/A decision are ADRs.
- **README = the graded report** (Phase 7): hardware + model justification, experiment, findings,
  theory-linking, economics, reproduction, all figures embedded inline. Framed as "where does
  layer-streaming pay off?" (D12). **README numbers match repo state at every commit** — no
  aspirational claims.
- Honest-disclosure triad: `KNOWN_LIMITATIONS.md` (severity + fix sketch), `SELF_GRADE.md`,
  `COST.md`. Documented limitation → small/zero deduction; hidden bug a grader finds → severe.
- `docs/PROMPTS.md`: only entries describing **committed** work; truthful and continuous.
- **The doc↔repo gap is the one fatal failure mode.** Public docs never contradict the tree.

---

## 9. Process

- **Conventional Commits** (`feat/fix/docs/refactor/test/chore/ci`), one atomic concern each,
  imperative subject ≤ 72 chars, body explains why + references TODO ids.
- **Branch → PR → review → squash-merge.** Nothing reaches `main` except via an approved PR.
- **Cross-model PR review:** a separate model family reviews each PR and posts findings as comments
  before merge (a verifiable trace). See `docs/REVIEW_PROCESS.md`.
- **Ownership represented honestly** (D13), consistent with `git shortlog`: Imree drives and authors
  most commits; Eyal reviews. No fabricated authorship.
- Self-grade **conservative** (target 92–93, cap 95), justified line-by-line by documented limitations.

---

## 10. Anti-patterns banned

Aspirational README · mock classes shadowing real imports · `NotImplementedError` stubs on `main` ·
tests committed without implementation · single mass-commit · leaked filesystem paths (`/mnt/`,
`C:\Users\…`) · `"AI Agent"` in authors · prompt-log entries for uncommitted work · stale TODO ·
treating a lecturer's "for example" value as a locked spec · **committing model weights**.
