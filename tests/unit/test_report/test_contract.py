"""Tests for the README report contract (T7.1).

The contract is keyless and pure: given README *text*, ``audit`` returns the list
of missing required sections + figures-not-embedded-inline. The README is the
graded deliverable, so this guards that every mandated section and every committed
figure is actually present before a commit can claim "report done" (D11, §8).
"""

from __future__ import annotations

from airllm_bench.report.contract import REQUIRED_FIGURES, REQUIRED_SECTIONS, audit


def _compliant_readme() -> str:
    """A minimal README that satisfies the contract, built from the contract itself."""
    headings = "\n".join(f"## {anchor.title()} details" for anchor in REQUIRED_SECTIONS)
    images = "\n".join(f"![{fig}](figures/{fig})" for fig in REQUIRED_FIGURES)
    return f"# AirLLM Benchmark Report\n\n{headings}\n\n{images}\n"


def test_compliant_readme_has_no_problems() -> None:
    assert audit(_compliant_readme()) == []


def test_missing_section_heading_is_flagged() -> None:
    target = REQUIRED_SECTIONS[0]
    stripped = "\n".join(
        line for line in _compliant_readme().splitlines() if target not in line.lower()
    )
    problems = audit(stripped)
    assert any(target in problem for problem in problems)


def test_every_missing_figure_is_flagged() -> None:
    headings = "\n".join(f"## {anchor} details" for anchor in REQUIRED_SECTIONS)
    problems = audit(f"# Report\n\n{headings}\n")
    assert len(problems) >= len(REQUIRED_FIGURES)


def test_plain_link_does_not_count_as_an_embedded_figure() -> None:
    headings = "\n".join(f"## {anchor} details" for anchor in REQUIRED_SECTIONS)
    fig = REQUIRED_FIGURES[0]
    # A bare markdown link (no leading '!') must not satisfy "embedded inline".
    text = f"# Report\n\n{headings}\n\n[{fig}](figures/{fig})\n"
    assert any(fig in problem for problem in audit(text))


def test_section_match_is_case_insensitive() -> None:
    headings = "\n".join(f"## {anchor.upper()} DETAILS" for anchor in REQUIRED_SECTIONS)
    images = "\n".join(f"![x](figures/{fig})" for fig in REQUIRED_FIGURES)
    assert audit(f"# R\n\n{headings}\n\n{images}\n") == []
