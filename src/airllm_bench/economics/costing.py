"""API token-cost computation.

Lifted from HW2's ``budget.cost_of`` with the spend-ceiling enforcement dropped
(HW5 makes no live calls — this is pure paper math on token counts, per D6/D8).
Prices are quoted per million tokens (Mtok), matching how providers list them;
the actual numbers live in ``config/economics.json`` with a source + date.
"""

from __future__ import annotations

from pydantic import BaseModel

#: Provider prices are quoted in USD per this many tokens.
_TOKENS_PER_MTOK = 1_000_000


class TokenRate(BaseModel):
    """USD price per million tokens, input and output billed separately."""

    input_per_mtok: float
    output_per_mtok: float


def cost_api(prompt_tokens: int, completion_tokens: int, rate: TokenRate) -> float:
    """Return the USD cost of one API call, billing input and output separately.

    Example:
        >>> cost_api(1_000_000, 0, TokenRate(input_per_mtok=0.2, output_per_mtok=0.4))
        0.2
    """
    billed = prompt_tokens * rate.input_per_mtok + completion_tokens * rate.output_per_mtok
    return billed / _TOKENS_PER_MTOK


def cost_api_cached(
    cached_tokens: int,
    fresh_input_tokens: int,
    completion_tokens: int,
    rate: TokenRate,
    cached_discount: float,
) -> float:
    """USD cost when some input is served from a prompt cache at a discount (D6).

    ``cached_discount`` is the fraction of the input rate charged for cached tokens
    (e.g. 0.1 = cached input billed at 10% of the normal input rate).
    """
    cached = cached_tokens * rate.input_per_mtok * cached_discount
    fresh = fresh_input_tokens * rate.input_per_mtok
    out = completion_tokens * rate.output_per_mtok
    return (cached + fresh + out) / _TOKENS_PER_MTOK
