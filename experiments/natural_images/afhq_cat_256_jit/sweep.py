"""Enumerate and submit AFHQ cat 256x256 JiT sweep jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
MODULE_NAME = "experiments.natural_images.afhq_cat_256_jit.run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or submit the AFHQ cat 256x256 JiT sweep.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--account", default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from experiments.natural_images.common.runtime import (
        enumerate_runs,
        load_config_and_paths,
        repo_root_from_experiment_dir,
        save_manifest,
        submit_slurm,
        write_slurm_script,
    )

    config, paths, _resolved_paths_file = load_config_and_paths(
        EXPERIMENT_DIR,
        config_path=args.config,
        paths_path=args.paths,
    )
    runs = enumerate_runs(config)
    manifest_path = save_manifest(EXPERIMENT_DIR, MODULE_NAME, runs)

    account = args.account or paths.get("slurm_account", "berzelius-2025-NNN")
    project_dir = args.project_dir or paths.get(
        "cluster_project_dir",
        str(repo_root_from_experiment_dir(EXPERIMENT_DIR)),
    )
    script_path = write_slurm_script(
        EXPERIMENT_DIR,
        MODULE_NAME,
        manifest_path,
        len(runs),
        partition=args.partition,
        time_limit=args.time,
        account=account,
        project_dir=project_dir,
    )

    print(f"Experiment: {EXPERIMENT_DIR.name}")
    print(f"Runs: {len(runs)}")
    print(f"Manifest: {manifest_path}")
    print(f"SLURM script: {script_path}")
    if args.dry_run:
        print(json.dumps(runs[:2], indent=2))
        return
    submit_slurm(script_path)


if __name__ == "__main__":
    main()
