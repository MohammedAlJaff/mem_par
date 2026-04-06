# Sanity Checks

This folder contains small, interpretable scripts that answer a simple question before we trust larger experiments: "is the machinery wired correctly enough to be worth scaling up?"

Each script has a different teaching role:

- `check_3x3_grid.py`: manual, visual-first smoke test for the full prediction/loss grid. Good for intuition and eyeballing obvious failures.
- `check_memorization.py`: more structured 3x3 diagnostic with summary metrics and plots. Despite the filename, it is still heuristic and should not be treated as a proof of memorization.
- `check_unet.py`: image-model plumbing check. Verifies shapes, short-run optimization, and basic sampling for the UNet stack.
- `check_jit_memory.py`: capacity check for the JiT image model. Useful before launching expensive high-resolution sweeps.

Suggested reading and run order:

1. Start with `check_3x3_grid.py` if you want intuition for what the 3x3 grid is doing.
2. Run `check_memorization.py` when you want a more reproducible summary of the same idea.
3. Run `check_unet.py` before spending time debugging image experiments.
4. Run `check_jit_memory.py` before committing to 128x128 or 256x256 JiT sweeps on a specific GPU.

What a "pass" means here:

- The code can train/sample without obvious numerical or shape failures.
- The chosen parameterizations produce qualitatively sensible behavior.

What a "pass" does **not** mean:

- The implementation is mathematically proven correct.
- The model is good enough for paper-quality results.
- The system will scale cleanly to larger datasets, longer runs, or different hardware.
