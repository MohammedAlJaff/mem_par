"""
Replay checkpoint evaluation for saved intrinsic/ambient hypersphere runs.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intr_amb_core import parse_eval_args, replay_run


def main(argv=None):
    args = parse_eval_args(argv)
    run_dir = replay_run(
        run_dir=args.run_dir,
        device=args.device,
        print_eval=args.print_eval,
    )
    print(f"run_dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
