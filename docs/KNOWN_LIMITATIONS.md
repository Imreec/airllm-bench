# Known Limitations

Every accepted imperfection, disclosed up front. A documented limitation is cheap; a hidden one a
grader finds is fatal to credibility (COURSE_LESSONS §4). Severity: **P0** blocks submission · **P1**
serious but acceptable · **P2** minor.

| ID | Limitation | Severity | Why accepted | Fix sketch |
|----|-----------|----------|--------------|------------|
| L-01 | **Single-machine measurement.** Benchmarks run once on one box (RTX 3080 Ti / 32 GB); CI cannot reproduce them. | P1 | Inherent to the assignment — it measures *this* hardware. | Commit raw JSON + pinned env as evidence; the analysis pipeline (data→figures) *is* CI-reproducible (D11 two-tier). |
| L-02 | **Windows `psutil` exposes no system page-cache figure.** The page-cache signal is cold-vs-warm timing + `available`, not a cached-bytes number. | P2 | OS limitation surfaced by the G-SPIKE (ADR 0001). | Report cold/warm deltas + `available`; document the proxy explicitly. |
| L-03 | **CPU energy is TDP-estimated, not measured.** GPU energy is measured (NVML); Ryzen package power isn't cleanly scriptable on Windows. | P2 | RAPL access on Windows is impractical. | Declare the TDP assumption in `economics.json`; GPU (the dominant draw) is measured. |
| L-04 | **`transformers` pinned `<4.43`.** AirLLM's layer driver predates the 4.43 RoPE refactor; newer transformers crash mid-generation. | P2 | Hard AirLLM constraint (ADR 0001), not a free choice. | Revisit if AirLLM updates its layer-call API. |
| L-05 | **Perplexity as a quality proxy.** Output quality per quant level is measured by perplexity + a small sample table, not human eval. | P2 | Deterministic, reproducible, theory-linked; full human eval is out of scope. | Disclose; the sample-output table gives a human-readable check. |
| L-06 | **Tiny token budgets at FP16.** FP16 (~8 min/token) scenarios use very few output tokens. | P2 | Latency is the finding; large generations are infeasible. | Size per ADR 0001; report token counts alongside every timing. |

_(Add an entry the moment a real gap is accepted — never let one go undocumented.)_
