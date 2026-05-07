# Figure Regeneration

This repo tracks compact metric snapshots for the current paper figures:

```text
outputs/evals/mem_boundary_20260427_all_done_quality_20260501/*.csv
outputs/evals/mem_boundary_20260427_all_done_quality_20260501/*.jsonl
outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_20260503/*.csv
outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_20260503/*.jsonl
outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_delta_20260503/*.csv
outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_delta_20260503/*.jsonl
```

The tracked files are enough to regenerate the paper-analysis and plotting PDFs/PNGs.
Per-attempt metrics, generated PDFs/PNGs, checkpoint files, and `.pt` eval caches stay
ignored.

Use the project environment, then run:

```bash
MPLCONFIGDIR=/tmp/mpl-mempar \
python prototyping/plot_mem_boundary_20260427_quality_memorization.py
```

This writes legacy N-sweep figures to:

```text
prototyping/figures/mem_boundary_20260427_all_done_quality_20260501_figures/
```

For merged N-sweep plus capacity paper-objective summaries:

```bash
MPLCONFIGDIR=/tmp/mpl-mempar \
python prototyping/analyze_mem_boundary_quality_mem_frontier.py
```

This writes merged tables and figures to:

```text
prototyping/figures/paper_objective_quality_mem_frontier_20260503/
```
