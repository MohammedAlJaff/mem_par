"""Small synthetic end-to-end demo of the reorganized training pipeline.

Example:
    python lab/demo_pipeline.py --device cpu
    python lab/demo_pipeline.py --config experiments/synthetic/high_d/torus/config.yaml --d 4 --n 500
"""

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.synthetic.high_d.common.data import GMMSubspace, TorusEmbedding
from experiments.synthetic.high_d.common.flow_matching import FlowMatching, sample_logit_normal
from experiments.synthetic.high_d.common.networks import TimeConditionedMLP
from experiments.synthetic.high_d.common.runtime import load_yaml, resolve_run_config
from experiments.synthetic.high_d.common.training import Trainer
from experiments.synthetic.high_d.common.utils import seed_everything


def _reshape_t_like_batch(t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Broadcast (B,) timesteps to match vector or image batches."""
    while t.dim() < batch.dim():
        t = t.unsqueeze(-1)
    return t


def parse_args():
    parser = argparse.ArgumentParser(description="Demo the synthetic data/model/loss/training pipeline")
    parser.add_argument("--config", default="experiments/synthetic/high_d/gmm_subspace/config.yaml")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--pred-space", type=str, default=None)
    parser.add_argument("--loss-space", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--run-dir", default="lab/demo_runs/demo_pipeline")
    return parser.parse_args()


def build_dataset(config: dict, run_config: dict):
    data_cfg = config["data"]
    if "K" in data_cfg:
        return GMMSubspace(
            D=data_cfg["D"],
            d=run_config["d"],
            K=data_cfg.get("K", 10),
            sigma_z=data_cfg.get("sigma_z", 0.5),
            sigma_eta=data_cfg.get("sigma_eta", 1e-3),
            n=run_config["n"],
            geometry_seed=data_cfg.get("geometry_seed", 0),
            data_seed=run_config["seed"],
        )
    return TorusEmbedding(
        D=data_cfg["D"],
        d=run_config["d"],
        hidden_mult=data_cfg.get("hidden_mult", 4),
        n=run_config["n"],
        geometry_seed=data_cfg.get("geometry_seed", 0),
        data_seed=run_config["seed"],
    )


def build_network(config: dict, dataset):
    model_cfg = config["model"]
    return TimeConditionedMLP(
        input_dim=dataset[0].numel(),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        depth=model_cfg.get("depth", 3),
        time_embed_dim=model_cfg.get("time_embed_dim", 64),
        activation=model_cfg.get("activation", "silu"),
    )


def main():
    args = parse_args()
    config = load_yaml(args.config)
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

    seed_everything(run_config["seed"])

    dataset = build_dataset(config, run_config)
    network = build_network(config, dataset)

    flow_model = FlowMatching(
        network=network,
        pred_space=run_config["pred_space"],
        loss_space=run_config["loss_space"],
    ).to(args.device)

    sample = dataset[0]
    batch_size = min(args.batch_size, len(dataset))
    batch = torch.stack([dataset[i] for i in range(batch_size)]).to(args.device)

    t = sample_logit_normal(batch_size=batch.shape[0], device=args.device)
    eps = torch.randn_like(batch)
    t_broad = _reshape_t_like_batch(t.unsqueeze(1), batch)
    z_t = t_broad * batch + (1 - t_broad) * eps

    with torch.no_grad():
        raw_pred = network(z_t, t)
        initial_loss = flow_model.compute_loss(batch).item()

    print("=" * 72)
    print("PIPELINE DEMO")
    print("=" * 72)
    print(f"Config:        {args.config}")
    print(f"Run config:    {run_config}")
    print(f"Dataset kind:  {'gmm_subspace' if 'K' in config['data'] else 'torus'}")
    print(f"Dataset size:  {len(dataset)}")
    print(f"Sample shape:  {tuple(sample.shape)}")
    print(f"Batch shape:   {tuple(batch.shape)}")
    print(f"Noisy z_t:     {tuple(z_t.shape)}")
    print(f"Net output:    {tuple(raw_pred.shape)}")
    print(f"Pred space:    {run_config['pred_space']}")
    print(f"Loss space:    {run_config['loss_space']}")
    print(f"Parameters:    {sum(p.numel() for p in network.parameters()):,}")
    print(f"Initial loss:  {initial_loss:.6f}")

    if args.train_steps > 0:
        demo_config = copy.deepcopy(config)
        demo_config["training"] = dict(demo_config["training"])
        demo_config["training"]["batch_size"] = min(
            demo_config["training"].get("batch_size", batch_size), len(dataset)
        )
        demo_config["training"]["max_steps"] = args.train_steps
        demo_config["training"].pop("max_epochs", None)
        demo_config["training"]["ckpt_interval_steps"] = args.train_steps
        demo_config["training"].pop("ckpt_interval_epochs", None)

        trainer = Trainer(
            flow_model, dataset, demo_config, args.run_dir, device=args.device
        )
        summary = trainer.train()
        print(f"Train steps:   {summary['total_steps']}")
        print(f"Final loss:    {summary['final_loss']:.6f}")

        trainer.ema.apply()
        with torch.no_grad():
            sample_shape = (min(4, batch_size),) + tuple(sample.shape)
            generated = flow_model.sample(
                shape=sample_shape,
                n_steps=args.sample_steps,
                method=config["flow_matching"].get("sampler", "heun"),
                device=args.device,
            ).cpu()
        trainer.ema.restore()

        print(f"Generated:     {tuple(generated.shape)}")
        print(f"Gen mean/std:  {generated.mean():.4f} / {generated.std():.4f}")
        print(f"Run dir:       {args.run_dir}")

    print("\nHow the pieces connect:")
    print("  1. Dataset returns clean examples x.")
    print(
        "  2. FlowMatching samples t and noise eps, then builds z_t = t*x + (1-t)*eps."
    )
    print("  3. The network maps (z_t, t) -> prediction in pred_space.")
    print("  4. FlowMatching converts that prediction into loss_space and applies MSE.")
    print("  5. Trainer repeats that step, updates the optimizer, scheduler, and EMA.")


if __name__ == "__main__":
    main()
