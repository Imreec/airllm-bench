# Git Workflow

> Read this before making any commit. Both partners. GitHub Flow + branch protection for this repo.
> Conventions here match `CLAUDE.md §9` and the `commit-discipline` / `pr-discipline` skills.

---

## One-time setup (per developer, per machine)

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"   # email tied to your GitHub account
git config --global pull.rebase true               # cleaner history
git config --global fetch.prune true               # auto-cleanup remote-tracking branches
gh auth login                                      # GitHub CLI

git clone https://github.com/Imreec/airllm-bench.git
cd airllm-bench
uv sync                 # core + dev deps
uv run pre-commit install
```

---

## Branch naming — `<type>/<short-name>`

Type-based, matching Conventional-Commit types and our actual history:

- `docs/<name>` — planning/docs work (`docs/prd`, `docs/plan`, `docs/todo`)
- `chore/<name>` — tooling/bootstrap (`chore/phase1-bootstrap`)
- `feat/<name>` — features (`feat/harness-sampler`, `feat/airllm-runner`)
- `fix/<name>`, `test/<name>`, `refactor/<name>`, `ci/<name>`
- `main` — protected, CI-green-only, PR-merged-only.

After merge, the branch is deleted (the `--delete-branch` flag handles it).

---

## Daily workflow

```bash
# Start fresh
git checkout main && git pull --ff-only
git checkout -b feat/harness-sampler

# Edit, then stage the specific files you changed
git add src/airllm_bench/harness/sampler.py tests/unit/test_harness/test_sampler.py

# Commit (pre-commit hooks run automatically; see commit-discipline skill)
git commit -m "feat(harness): ResourceSampler thread for NVML power + psutil RAM"

# First push
git push --set-upstream origin feat/harness-sampler
# Subsequent pushes: git push
```

Commit messages end with the agent co-author trailer when the agent did the work:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
For Eyal's carried-over work: `Co-Authored-By: Eyal Shtinmetz <eyalshtinmetz@gmail.com>`.

---

## Opening a PR

```bash
gh pr create --base main --head feat/harness-sampler
```
The PR template loads — fill every section. Then the `pr-discipline` flow: wait for CI, address the
cross-model review on the thread, sync README/TODO.

---

## Your branch is behind main (someone merged first)

```bash
git checkout main && git pull --ff-only
git checkout feat/harness-sampler
git rebase main                  # resolve conflicts: edit, git add, git rebase --continue
git push --force-with-lease      # NEVER plain --force; NEVER force-push main
```

---

## Merging — squash only

When the PR is ready and CI is green, **squash-merge** (one meaningful commit per concern; clean
linear history — this is what our `git log` already shows):

```bash
gh pr merge <N> --squash --delete-branch
git checkout main && git pull --ff-only
```

We do **not** use merge-commits or rebase-merge — squash keeps `main` a clean sequence of
one-commit-per-PR. (This is a deliberate departure from some GitHub-Flow guides.)

---

## Common gotchas

| Symptom | Fix |
|---------|-----|
| `Updates were rejected … remote contains work` | `git pull --rebase` then push |
| Branch behind main | rebase on main (above) |
| `pre-commit hook failed` | read output, fix, re-stage, re-commit |
| Accidentally committed to `main` | pre-commit blocks it; if past it: `git reset HEAD~1`, branch, re-commit |
| Committed a secret / model weight | remove from history (`git filter-repo`), rotate the secret |
| PR shows a huge diff | you forgot to rebase — rebase locally, force-with-lease |

## When in doubt

`git status` · `git branch --show-current` · `git log --oneline -10` · `git diff [--staged]` ·
`make grade` — these answer 90% of "wait, what just happened?"

---

## Running parallel Claude Code sessions (worktrees)

Don't run two sessions on different branches from the same directory — they trample each other's
files and test runs. Use git worktrees: each is a separate working directory on a different branch
sharing one `.git`. Prefer Claude Code's `-w` flag.

```bash
claude -w feat-airllm-runner      # worktree under .claude/worktrees/ on a new branch + a session
claude -w feat-economics          # a second, fully isolated parallel session
```

One-time repo setup (already done at bootstrap):
- `.claude/worktrees/` is git-ignored.
- `.worktreeinclude` lists git-ignored files (notably `.env` with `HF_TOKEN`) copied into each new
  worktree — without it a worktree session can't download models.

Conventions: run `uv sync` once per fresh worktree; don't share a venv across worktrees; don't run
hardware/Tier-2 work in parallel worktrees (they'd contend on the single GPU). Cleanup:
`git worktree list` / `git worktree remove <path>` / `git worktree prune` — prune routinely.

### When not to bother
Small sequential tasks; working alone on one branch; anything contending on the single GPU.
