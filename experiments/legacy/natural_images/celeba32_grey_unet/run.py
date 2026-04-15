"""Run one CelebA 32x32 greyscale UNet experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def build_dataset(config: dict, run_config: dict, paths: dict):
    from experiments.natural_images.common.data import CelebA32Grey

    data_root = paths.get("data_root")
    if not data_root:
        raise ValueError("Set data_root in paths.local.yaml before running CelebA experiments.")
    data_cfg = config["data"]
    return CelebA32Grey(
        root=data_root,
        split=data_cfg.get("split", "train"),
        n=run_config["n"],
        seed=run_config["seed"],
    )


def build_network(config: dict, run_config: dict, dataset):
    from experiments.natural_images.common.networks import SmallUNet

    model_cfg = config["model"]
    return SmallUNet(
        in_channels=model_cfg.get("in_channels", 1),
        base_channels=model_cfg.get("base_channels", 64),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 4])),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [8])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        dropout=model_cfg.get("dropout", 0.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CelebA 32x32 greyscale UNet experiment.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manifest", default=str(EXPERIMENT_DIR / "runs" / "run_manifest.json"))
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--pred-space", default=None)
    parser.add_argument("--loss-space", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from experiments.natural_images.common.runtime import (
        execute_image_run,
        load_config_and_paths,
        load_run_from_manifest,
        resolve_run_config,
    )

    config, paths, _resolved_paths_file = load_config_and_paths(
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
                "n": args.n,
                "pred_space": args.pred_space,
                "loss_space": args.loss_space,
                "seed": args.seed,
            },
        )

    execute_image_run(
        EXPERIMENT_DIR,
        config,
        run_config,
        build_dataset=build_dataset,
        build_network=build_network,
        device=args.device,
        paths=paths,
    )


if __name__ == "__main__":
    main()
