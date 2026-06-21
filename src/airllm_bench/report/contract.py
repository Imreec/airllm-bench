"""The README contract — what the graded report MUST contain (D11, §8).

The README *is* the deliverable, so this encodes the floor a grader expects to
find top-down: every mandated section heading and every committed figure embedded
*inline* (a bare link doesn't count — figures must render in the report). Pure and
keyless: ``audit`` takes README text and returns the list of gaps, empty when clean.

Kept in lockstep with ``check_raw_data._EXPECTED_FIGURES`` (the figure set) and the
section structure the report is written to.
"""

from __future__ import annotations

import re

#: Every figure `make figures` emits must be embedded inline in the report.
REQUIRED_FIGURES: tuple[str, ...] = (
    "ttft_vs_length.png",
    "itl.png",
    "cold_warm_ttft.png",
    "throughput.png",
    "perplexity.png",
    "roofline.png",
    "breakeven.png",
)

#: Lowercased substrings that must each appear in at least one heading. These are
#: the mandated report sections (PRD §5 deliverables + the spec-compliance table).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "compliance",  # the spec-compliance / requirement-coverage table, led with
    "hardware",  # hardware spec + the OOM-wall justification
    "model",  # model-choice justification (Qwen2.5-32B)
    "scenario",  # the experiment / scenario matrix
    "finding",  # measured findings
    "theory",  # theory-linking (prefill/decode, memory/compute-bound, paging)
    "roofline",  # the hierarchical-roofline extension
    "economic",  # break-even economics
    "reproduc",  # reproduction instructions
    "limitation",  # known limitations + self-grade
)

#: Markdown image embed: ``![alt](target)``. The leading ``!`` is what separates an
#: embedded (rendered) figure from a plain link.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]*)\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _heading_texts(text: str) -> list[str]:
    """Lowercased text of every markdown heading line."""
    return [match.lower() for match in _HEADING_RE.findall(text)]


def _embedded_image_targets(text: str) -> list[str]:
    """The target (URL/path) of every ``![...](target)`` image embed."""
    return _IMAGE_RE.findall(text)


def audit(text: str) -> list[str]:
    """Return every contract gap in ``text`` (empty list ⇒ the report is complete)."""
    problems: list[str] = []
    headings = _heading_texts(text)
    for anchor in REQUIRED_SECTIONS:
        if not any(anchor in heading for heading in headings):
            problems.append(f"missing required section heading for '{anchor}'")
    targets = _embedded_image_targets(text)
    for figure in REQUIRED_FIGURES:
        if not any(figure in target for target in targets):
            problems.append(f"figure not embedded inline: {figure}")
    return problems
