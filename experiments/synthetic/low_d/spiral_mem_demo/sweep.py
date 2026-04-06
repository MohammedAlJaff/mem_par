"""Lightweight sweep helper for the spiral memorization demo."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit spiral-demo run commands from config.yaml.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from experiments.synthetic.low_d.common.runtime import load_config_and_paths

    config, _paths, _resolved_paths_file = load_config_and_paths(
        EXPERIMENT_DIR,
        config_path=args.config,
        paths_path=args.paths,
    )
    sweep = config.get("sweep", {})
    axes = {
        "n": sweep.get("n", [config["defaults"]["n"]]),
        "pred_space": sweep.get("pred_space", [config["defaults"]["pred_space"]]),
        "seed": sweep.get("seed", [config["defaults"]["seed"]]),
    }
    commands = []
    for n_value, pred_space, seed in itertools.product(axes["n"], axes["pred_space"], axes["seed"]):
        commands.append(
            [
                "python",
                "-m",
                "experiments.synthetic.low_d.spiral_mem_demo.run",
                "--n",
                str(n_value),
                "--pred-space",
                pred_space,
                "--seed",
                str(seed),
            ]
        )

    print(json.dumps(commands, indent=2))


if __name__ == "__main__":
    main()
