# Review Process

> How code reaches `main`: terminal authoring → automated quality gates → independent cross-model
> review → human adjudication.

## Pipeline

1. **Authoring (terminal).** Code is written via the Claude Code CLI (`CLAUDE.md §9`). An IDE may
   serve as editor/shell; there is no IDE-coupled inline AI completion.
2. **Automated gates (CI + pre-commit).** Every commit and PR runs `ruff`, `mypy --strict`, `pytest`
   with ≥90% coverage on the deterministic core, the ≤150-line file-size check, and the
   anti-pattern / no-hardcoded scanners. Nothing merges red (`CLAUDE.md §1`).
3. **Independent cross-model review.** Each pull request is reviewed by a **separate model family**
   (different from the Claude author), which posts its findings **as PR comments** — a verifiable
   trace on the PR itself.
4. **Human adjudication.** Imree (driver) and Eyal (reviewer) read the review, accept or push back on
   each finding with reasoning, and address accepted ones with follow-up commits on the same branch.
   Each resolution is replied to on the thread.

## Why a different model reviews

Author and reviewer are **different models**, so the reviewer is not anchored to the author's
assumptions and brings different blind spots — the same rationale that makes independent peer review
standard. It does not replace human judgement; the humans make the final call (an approval = the gate).

## On the record (this repo)

The planning gates went through this pipeline and the cross-model reviewer caught real issues, all
resolved in follow-up commits visible in the PR comment history:

- **PR #1 (PRD):** the baseline FP16 OOM was vulnerable to `accelerate` auto-offload under
  `device_map="auto"` (would defeat the clean OOM); roofline NF4 ceiling refined.
- **PR #2 (PLAN):** a *blocking* find — an in-process length×phase sweep would warm the OS page cache
  and corrupt every later "cold" reading; restructured to one scenario per `RunResult` with the matrix
  orchestrated externally. Plus native per-runtime tokenization for valid perplexity.
- **PR #3 (TODO):** a *blocking* WSL2 memory-cap find (default ~50% host RAM can't cache the model →
  thrash); plus a fail-loud privilege guard for the Windows page-cache flush.

Both blocking finds were silent-corruption failure modes — data invalidated without a crash — which is
exactly the class this independent review exists to catch.
