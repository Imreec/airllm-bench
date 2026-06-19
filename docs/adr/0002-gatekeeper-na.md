# ADR 0002 — API Gatekeeper is N/A for this project

- **Status:** Accepted
- **Date:** 2026-06-19
- **Context:** PRD D8 · CLAUDE.md §3 · the HW2/HW3 "omit-if-no-raw-calls" precedent.

## Context

The course standards (and our `self_grade` drift check) expect every external API call to route
through a single `ApiGatekeeper` (rate-limit, retry, log). HW3 omitted it by ADR because CrewAI owned
dispatch. HW5 is different again, so we record the decision explicitly rather than leave a missing
module to be mis-read as an oversight (COURSE_LESSONS §12: reconcile, don't silently skip).

## Decision

**No `ApiGatekeeper` is built. HW5 makes zero live third-party API calls.**

- **Economics (D6/D8):** the API side of the cost analysis is *paper math on token counts*, not real
  requests. Prices are documented constants in `config/economics.json` (each with source + date);
  token counts come from the local tokenizer. There is no live pricing pull and no inference API call.
  `.env` carries only `HF_TOKEN` (model downloads).
- **Local inference** (the measured subject) runs through AirLLM / transformers / llama.cpp and lives
  **openly in `runners/`** — it is not an external API to be gatekept; it *is* the experiment.

So CLAUDE.md's "API calls outside the Gatekeeper: 0" is satisfied **vacuously** — there are no such
calls. Building a Gatekeeper would be dead code, which the standards explicitly discourage.

## Consequences

- The self-grade drift check must treat a missing Gatekeeper as **expected** for this project (point
  it at this ADR).
- If a live third-party call is ever added (it should not be), this ADR must be revisited and a
  minimal Gatekeeper introduced.

## Alternatives considered

- **A minimal Gatekeeper for a live OpenRouter pricing pull** — rejected (PRD D8): a live price is
  *less* reproducible than a cited constant, breaks keyless-by-default, and adds a key + network
  dependency for a number we can write down once with a source.
