"""Run one AFHQ cat 256x256 JiT experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def build_dataset(config: dict, run_config: dict, paths: dict):
    from experiments.natural_images.common.data import ImageFolderSubset

    data_root = paths.get("data_root")
    if not data_root:
        raise ValueError("Set data_root in paths.local.yaml before running AFHQ experiments.")
    data_cfg = config["data"]
    return ImageFolderSubset(
        root=data_root,
        resolution=data_cfg.get("resolution", 256),
        crop_mode=data_cfg.get("crop_mode", "resize"),
        n=run_config["n"],
        seed=run_config["seed"],
    )


def build_network(config: dict, run_config: dict, dataset):
    from experiments.natural_images.common.networks import JiTWrapper

    model_cfg = config["model"]
    return JiTWrapper(
        image_size=config["data"].get("resolution", 256),
        patch_size=model_cfg.get("patch_size", 16),
        in_channels=model_cfg.get("in_channels", 3),
        hidden_dim=model_cfg.get("hidden_dim", 768),
        depth=model_cfg.get("depth", 12),
        num_heads=model_cfg.get("num_heads", 12),
        bottleneck_dim=model_cfg.get("bottleneck_dim", 128),
        dropout=model_cfg.get("dropout", 0.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AFHQ cat 256x256 JiT experiment.")
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
