# Self-Grade

> **Skeleton — finalized at submission** (after the report, all runs, and `make grade` are green).
> Computed by `scripts/self_grade.py` and ratified by both partners. Target **92–93**, cap 95. The
> number is meaningless without the per-category justification below; `KNOWN_LIMITATIONS.md` *is* the
> justification for the gap below 100 (COURSE_LESSONS §3).

## Rubric (analysis-weighted — the brief's emphasis)

| Category | Weight | Score | Justification |
|----------|--------|-------|---------------|
| Analysis & theory-linking | ~35 | _TBD_ | baseline/AirLLM/quant/competitor depth; Prefill/Decode + memory/compute-bound + paging; the roofline |
| Measurement rigor & reproducibility | ~25 | _TBD_ | six metrics captured cleanly; cold/warm; committed raw data; figures regenerate offline |
| Economics | ~15 | _TBD_ | two on-prem lines + API + caching + cloud; break-even; assumptions stated |
| Report / README quality | ~15 | _TBD_ | hardware + model justification; findings; reproduction; figures embedded |
| Process & code quality | ~10 | _TBD_ | ruff/mypy/coverage/file-size; atomic commits; cross-model review; honest disclosure |
| **Total** | **100** | **_TBD_** | |

## Honesty notes (to apply when filling this in)
- Report **below** honest belief; if torn between two numbers, take the lower.
- Never claim > 95 without an explicit audit that lowers at least one judgment-call item.
- A well-analyzed **negative** result (AirLLM dominated by llama.cpp) scores *higher* than a shallow
  positive one — do not treat it as a deduction.
