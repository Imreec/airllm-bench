# Self-Grade

> **Final.** Computed from a data-driven rubric ([`config/self_grade.json`](../config/self_grade.json))
> and re-runnable: `uv run python scripts/self_grade.py`. Ratified by both partners. The number is
> meaningless without the per-category justification below; [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)
> *is* the justification for the gap below 100 (COURSE_LESSONS §3).

## Score: **92.8 / 100** (cap 95)

Analysis-weighted, matching the brief's emphasis. Each category is scored 0–100 and weighted by its
share; the weighted total is what `scripts/self_grade.py` prints (and the CI gate validates as ≤ cap).

| Category | Weight | Score | Weighted | Justification |
|----------|-------:|------:|---------:|---------------|
| Analysis & theory-linking | 35 | 93 | 32.55 | Baseline OOM → AirLLM quant sweep → llama.cpp competitor, all linked to theory (prefill/decode, memory/compute-bound, VRAM wall, paging). The hierarchical roofline ties measured points to ceilings. **Gap:** roofline tiers are assigned by data path, not measured saturation ([L-10](KNOWN_LIMITATIONS.md)). |
| Measurement rigor & reproducibility | 25 | 93 | 23.25 | Six metrics from per-token streaming timestamps (never total ÷ count); paired cold/warm protocol; committed raw JSON as the source of truth; figures regenerate offline & in CI. **Gap:** single machine, cold-flush not magnitude-verified on Windows ([L-01](KNOWN_LIMITATIONS.md), [L-07](KNOWN_LIMITATIONS.md)). |
| Economics | 15 | 91 | 13.65 | Five lines (API · API+cache · on-prem realistic · on-prem AirLLM · cloud); break-even; every constant sourced + dated; on-prem electricity from **measured** energy. **Gap:** anchors on near-comparable SKUs, not identical ([L-09](KNOWN_LIMITATIONS.md)). |
| Report / README quality | 15 | 92 | 13.80 | Spec-compliance table, hardware + model justification, all 7 figures embedded inline, theory, reproduction, dedicated engineering section; every number a function of committed data. **Gap:** prose is dense; some figures (roofline) remain visually busy despite the de-clutter. |
| Process & code quality | 10 | 95 | 9.50 | `ruff`/`mypy --strict`/≥90% coverage (actual 99%)/≤150-line files all CI-enforced; atomic squash-merged PRs; cross-model review on every PR; honest-disclosure triad + a runnable self-grade. |
| **Total** | **100** | | **92.75** | rounds to **92.8** |

## Honesty notes (applied)
- Reported **below** honest belief; where torn between two numbers, took the lower (e.g. Economics 91
  for the SKU-comparability caveat, not 93).
- No category claims > 95; the cap is never approached without a judgment-call deduction already taken.
- The **negative result** (AirLLM dominated by llama.cpp on throughput and cost) is scored as a
  *well-characterized finding*, not a deduction — it is the report's thesis, evidenced and theory-linked
  (D12). A shallow positive result would score lower.
- Every gap above maps to a documented entry in [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md); none is
  hidden. The 7.2-point gap below 100 is the sum of those disclosed, accepted imperfections.
