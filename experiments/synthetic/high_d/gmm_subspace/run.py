"""Run one GMM-subspace synthetic experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def build_dataset(config: dict, run_config: dict):
    from experiments.synthetic.high_d.common.data import GMMSubspace

    data_cfg = config["data"]
    return GMMSubspace(
        D=data_cfg["D"],
        d=run_config["d"],
        K=data_cfg.get("K", 10),
        sigma_z=data_cfg.get("sigma_z", 0.5),
        sigma_eta=data_cfg.get("sigma_eta", 1.0e-3),
        n=run_config["n"],
        geometry_seed=data_cfg.get("geometry_seed", 0),
        data_seed=run_config["seed"],
    )


def build_network(config: dict, run_config: dict, dataset):
    from experiments.synthetic.high_d.common.networks import TimeConditionedMLP

    model_cfg = config["model"]
    return TimeConditionedMLP(
        input_dim=dataset[0].numel(),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        depth=model_cfg.get("depth", 3),
        time_embed_dim=model_cfg.get("time_embed_dim", 64),
        activation=model_cfg.get("activation", "silu"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic GMM-subspace experiment.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manifest", default=str(EXPERIMENT_DIR / "runs" / "run_manifest.json"))
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--pred-space", default=None)
    parser.add_argument("--loss-space", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from experiments.synthetic.high_d.common.runtime import (
        execute_synthetic_run,
        load_config_and_paths,
        load_run_from_manifest,
        resolve_run_config,
    )

    config, _paths, _resolved_paths_file = load_config_and_paths(
        EXPERIMENT_DIR,
        config_path=args.config,
        paths_path=args.paths,
    )

    if args.run_index is not None:
        run_config = load_run_from_manifest(args.manifest, args.run_index)
    else:
        run_config = resolve_run_config(
            config,
            {
                "d": args.d,
                "n": args.n,
                "pred_space": args.pred_space,
                "loss_space": args.loss_space,
                "seed": args.seed,
            },
        )

    execute_synthetic_run(
        EXPERIMENT_DIR,
        config,
        run_config,
        build_dataset=build_dataset,
        build_network=build_network,
        device=args.device,
    )


if __name__ == "__main__":
    main()
