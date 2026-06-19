## Summary

<!-- One paragraph: what does this PR do, in user-facing terms? -->


## Linked Planning Items

<!-- References to PRD section, TODO items, related issues -->
- Implements PRD §
- Closes TODO items:
- Related: #


## Changes

<!-- Bulleted list of concrete changes -->
-


## Checklist (author — correctness)

- [ ] Tests pass locally (`make test`)
- [ ] Ruff clean (`make lint`)
- [ ] Mypy clean
- [ ] No file exceeds 150 lines
- [ ] No secrets / hardcoded values introduced
- [ ] README updated if user-visible behavior changed
- [ ] TODO items checked off
- [ ] If figures changed: regenerated and committed to `assets/`
- [ ] Conventional commit format
- [ ] Pre-commit hooks ran clean
- [ ] PROMPTS.md updated with the prompt that drove this work


## Checklist (author — self-review for quality)

Read your own diff in the GitHub PR view as if reviewing someone else's code. Verify:

- [ ] Code matches the linked PRD specification
- [ ] No `NotImplementedError` placeholders
- [ ] No mock classes shadowing real imports
- [ ] No `/mnt/`, `/tmp/`, or `~/` paths in committed files
- [ ] Tests cover edge cases, not just the happy path
- [ ] Docstrings on every new public function/class
- [ ] CI is green (status check passing)
