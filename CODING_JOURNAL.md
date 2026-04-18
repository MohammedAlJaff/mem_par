# Coding Journal

This journal is the shared session log for the repo. Use UTC ISO timestamps,
keep entries practical, and record enough context to resume work without
rereading the whole codebase.

## 2026-04-18T05:56:33Z - Starter State And Objectives

### Current State
- Repo path: `/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par`
- Branch state: `main`, ahead of `origin/main` by 1 commit.
- Latest commit: `742d759 Rename synthetic prototype runner and extract metrics`.
- Active IDE file: `ram_test.py`.
- Dirty files to preserve:
  - modified `experiments/synthetic/gmm_data_sanity.ipynb`
  - untracked `.codex`
  - untracked `ram_test.py`
  - untracked `prototyping/projected_gmm_metric_validation.ipynb`

### Research Objective
- Under fixed architecture, data, and training protocol, study how `x`, `eps`,
  and `v` target parameterizations change memorization dynamics.
- Key measurements: `f_mem`, nearest-neighbor ratio and distance statistics,
  and checkpoint-level memorization trajectories.

### Immediate Working Objective
- Track where we are across sessions.
- Record decisions, commands, results, blockers, and next actions.
- Make it easier to resume without losing context around synthetic GMM
  validation, notebook state, RAM/GPU memory probing, and
  canonical-vs-prototype migration work.

### Next Action
- Add this journal file.
- Add a short `AGENT.md` workflow note requiring future meaningful work to
  append or update a timestamped journal entry.
- Leave unrelated dirty notebook/test files untouched.

## Entry Template

```markdown
## YYYY-MM-DDTHH:MM:SSZ - Short Title

### Context
What part of the repo/workstream this entry belongs to.

### Intent
What we are trying to decide, implement, test, or understand.

### Changed / Observed
Files touched, facts learned, or behavior noticed.

### Commands / Checks
Commands run and what they were meant to verify.

### Results
Outcome, metrics, failures, or confidence level.

### Blockers
Anything unresolved or risky.

### Next
The next concrete action.
```
