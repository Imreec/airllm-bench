# Cost Ledger

## Live API spend: $0.00

HW5 is **keyless and makes no paid API calls** (D8, ADR 0002). The "API cost" in this project is a
*modeled* quantity — paper math on token counts against documented price constants — not real spend.
So there is no attributed-vs-dashboard reconciliation to do here, unlike a live-LLM project.

## Incidental costs (non-USD)
- **Model download:** ~62 GB one-time over the HF Hub (bandwidth, not billed).
- **Electricity:** the on-prem benchmark runs draw real power; this is *measured* (GPU via NVML) and
  *estimated* (CPU via TDP) and feeds the economic analysis — see the README economics section and
  `config/economics.json`.

## Modeled costs (the analysis)
The full On-Prem vs API vs Cloud break-even — with every assumption parametrized in
`config/economics.json` — is produced in Phase 6 and presented in the README. This file exists to make
the **zero live spend** explicit and to point at where the modeled costs live.
