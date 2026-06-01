"""
Train one flow-matching parameterization on intrinsic/ambient ManifoldGMM data.

This is intentionally a single-run script. Sweeps over `x`, `eps`, and `v`
should happen in the job layer by launching this script once per `--param`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# Let this prototype be run directly from the repo root without installing the
# package. Job scripts can still set PYTHONPATH=src explicitly.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gdfm import GDFM
from mp_artifacts import save_checkpoint
from mp_config import config_to_dict, get_base_config
from mp_metrics.metrics_synthetic import compute_memorization_metrics

from data import ManifoldGMM, PARAMS


# =============================================================================
# CLI defaults and metric schema
# =============================================================================

DEFAULT_OUT_DIR = "prototyping/intr_amb_experiments/outputs/param_runs"

# The CSV and JSONL files use one row per evaluated checkpoint. Keeping a fixed
# field order makes comparison across job-launched x/eps/v runs simple.
METRIC_FIELDS = (
    "param",
    "step",
    "loss",
    "f_mem",
    "mean_nn_ratio",
    "std_nn_ratio",
    "mean_d1",
    "mean_d2",
    "gmm_nll",
    "gmm_nll_per_dim",
    "baseline_f_mem",
    "baseline_mean_nn_ratio",
    "baseline_std_nn_ratio",
    "baseline_mean_d1",
    "baseline_mean_d2",
    "baseline_gmm_nll",
    "baseline_gmm_nll_per_dim",
    "delta_f_mem",
    "delta_mean_nn_ratio",
    "delta_gmm_nll",
    "delta_gmm_nll_per_dim",
    "sample_count",
    "rho",
)


def parse_args(argv=None):
    """Read one single-run experiment specification from the command line."""
    parser = argparse.ArgumentParser(
        description="Train one FM parameterization on ManifoldGMM data."
    )
    parser.add_argument(
        "--param", required=True, choices=PARAMS, help="target parameterization"
    )
    parser.add_argument(
        "--D", "--ambient_dim", dest="D", type=int, default=8, help="ambient dimension"
    )
    parser.add_argument(
        "--K",
        "--intrinsic_dim",
        dest="K",
        type=int,
        default=2,
        help="intrinsic dimension",
    )
    parser.add_argument(
        "--M", "--n_modes", dest="M", type=int, default=5, help="number of GMM modes"
    )
    parser.add_argument(
        "--N",
        "--n_train",
        dest="N",
        type=int,
        default=100,
        help="number of train samples",
    )
    parser.add_argument(
        "--N_test",
        "--n_test",
        dest="N_test",
        type=int,
        default=1000,
        help="number of held-out reference samples",
    )
    parser.add_argument(
        "--mode_sigma",
        "--sigma",
        dest="mode_sigma",
        type=float,
        default=0.5,
        help="within-mode std along the projected manifold; --sigma is a legacy alias",
    )
    parser.add_argument(
        "--off_manifold_sigma",
        type=float,
        default=0.05,
        help="off-manifold data noise std; controls synthetic data thickness",
    )
    parser.add_argument(
        "--sep", type=float, default=2.0, help="GMM mode separation scale"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geometry_seed", type=int, default=0)
    parser.add_argument(
        "--tau",
        "--steps",
        dest="tau",
        type=int,
        default=50000,
        help="number of optimizer steps",
    )
    parser.add_argument(
        "--B", "--batch_size", dest="B", type=int, default=64, help="minibatch size"
    )
    parser.add_argument(
        "--H", "--hidden_dim", dest="H", type=int, default=256, help="MLP hidden width"
    )
    parser.add_argument(
        "--L",
        "--n_layers",
        dest="L",
        type=int,
        default=4,
        help="number of MLP hidden layers",
    )
    parser.add_argument(
        "--T_emb",
        "--time_emb_dim",
        dest="T_emb",
        type=int,
        default=64,
        help="time embedding dimension",
    )
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--rho", type=float, default=0.3, help="NN-ratio memorization threshold"
    )
    parser.add_argument(
        "--N_gen",
        "--sample_count",
        dest="N_gen",
        type=int,
        default=1000,
        help="number of generated samples per checkpoint",
    )
    parser.add_argument(
        "--tau_sample",
        "--sampling_steps",
        dest="tau_sample",
        type=int,
        default=128,
        help="ODE solver steps for generation",
    )
    parser.add_argument("--sampling_method", choices=("euler", "heun"), default="heun")
    parser.add_argument(
        "--tau_eval",
        "--ckpt_every",
        dest="tau_eval",
        type=int,
        default=5000,
        help="checkpoint/evaluation interval in optimizer steps",
    )
    parser.add_argument(
        "--tau_ckpts",
        "--ckpt_steps",
        dest="tau_ckpts",
        default="",
        help="explicit comma-separated checkpoint steps",
    )
    parser.add_argument(
        "--tau_log",
        "--log_every",
        dest="tau_log",
        type=int,
        default=100,
        help="loss logging interval in optimizer steps",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--print_eval", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args(sys.argv[1:] if argv is None else list(argv))


def resolve_device(name):
    """Resolve `auto` to the best available torch device."""
    if str(name) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(name)


def validate_args(args):
    """Fail early on settings that would make the experiment ill-defined."""
    checks = [
        (args.D >= 1, "--D must be at least 1."),
        (args.K >= 1, "--K must be at least 1."),
        (args.K <= args.D, "--K must be <= --D."),
        (args.M >= 1, "--M must be at least 1."),
        (args.N >= 2, "--N must be at least 2."),
        (args.N_test >= 2, "--N_test must be at least 2."),
        (args.mode_sigma > 0.0, "--mode_sigma must be positive."),
        (args.off_manifold_sigma > 0.0, "--off_manifold_sigma must be positive."),
        (args.tau >= 1, "--tau must be at least 1."),
        (args.B >= 1, "--B must be at least 1."),
        (args.N_gen >= 1, "--N_gen must be at least 1."),
        (args.tau_sample >= 1, "--tau_sample must be at least 1."),
        (args.tau_eval >= 1, "--tau_eval must be at least 1."),
        (args.tau_log >= 1, "--tau_log must be at least 1."),
        (args.chunk_size >= 1, "--chunk_size must be at least 1."),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def checkpoint_steps(args):
    """Return the exact optimizer steps where we save and evaluate checkpoints."""
    if str(args.tau_ckpts).strip():
        # Explicit comma-separated steps are useful for matching a job manifest.
        steps = {
            int(item.strip()) for item in str(args.tau_ckpts).split(",") if item.strip()
        }
    else:
        # Otherwise use a simple interval schedule.
        steps = set(range(0, int(args.tau) + 1, int(args.tau_eval)))
    steps.add(0)
    steps.add(int(args.tau))
    return tuple(sorted(step for step in steps if 0 <= int(step) <= int(args.tau)))


def make_run_dir(args):
    """Create the local artifact layout for this one parameterization run."""
    if args.run_id:
        name = args.run_id
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = (
            f"param-{args.param}_D-{args.D}_K-{args.K}_M-{args.M}_"
            f"N-{args.N}_Ntest-{args.N_test}_modesigma-{args.mode_sigma:g}_"
            f"offmanifoldsigma-{args.off_manifold_sigma:g}_"
            f"tau-{args.tau}_taueval-{args.tau_eval}_Ngen-{args.N_gen}_"
            f"seed-{args.seed}_gseed-{args.geometry_seed}_{stamp}"
        )
    run_dir = Path(args.out_dir).expanduser() / name
    for subdir in ("artifacts", "checkpoints", "metrics", "figures", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_dir


# =============================================================================
# Experiment construction
# =============================================================================


def build_config(args):
    """Build the minimal GDFM config needed by the src model wrapper."""
    config = get_base_config()
    config.study.name = "intr_amb_param_single"
    config.study.description = "Single ManifoldGMM parameterization prototype."
    config.study.dataset_key = "intr_amb_manifold_gmm"
    config.outputs.run_name = str(args.run_id or "auto")
    config.framework.family = "fm"
    # This script trains exactly one parameterization. External job scripts
    # should launch separate runs for x, eps, and v.
    config.framework.prediction_space = str(args.param)
    config.framework.loss_space = str(args.param)
    config.data.dataset = "intr_amb_manifold_gmm"
    config.data.data_dim = int(args.D)
    config.data.ambient_dim = int(args.D)
    config.data.intrinsic_dim = int(args.K)
    config.data.intrinsic_data_dim = int(args.K)
    config.data.n_train = int(args.N)
    config.data.n_test = int(args.N_test)
    config.data.n_clusters = int(args.M)
    config.data.sigma = float(args.mode_sigma)
    config.data.mode_sigma = float(args.mode_sigma)
    config.data.off_manifold_sigma = float(args.off_manifold_sigma)
    config.data.sep = float(args.sep)
    config.data.geometry_seed = int(args.geometry_seed)
    config.network.arch = "mlp"
    config.network.hidden_dim = int(args.H)
    config.network.n_layers = int(args.L)
    config.network.time_emb_dim = int(args.T_emb)
    config.training.seed = int(args.seed)
    config.training.n_training_steps = int(args.tau)
    config.training.batch_cap = int(args.B)
    config.training.batch_size = int(args.B)
    config.training.lr = float(args.lr)
    config.training.weight_decay = float(args.weight_decay)
    config.training.optimizer = str(args.optimizer)
    config.training.momentum = float(args.momentum)
    config.training.ckpt_every = int(args.tau_eval)
    config.training.ckpt_steps = tuple(checkpoint_steps(args))
    config.training.resolved_ckpt_steps = tuple(checkpoint_steps(args))
    config.training.log_every = int(args.tau_log)
    config.evaluation.sample_bank_size = int(args.N_gen)
    config.evaluation.n_samples = int(args.N_gen)
    config.evaluation.batch_size = int(args.B)
    config.evaluation.ratio_threshold = float(args.rho)
    config.evaluation.chunk_size = int(args.chunk_size)
    config.sampling.method = str(args.sampling_method)
    config.sampling.n_steps = int(args.tau_sample)
    return config


def build_dataset(args):
    """Build one fixed train/reference pair from the ManifoldGMM prototype."""
    # Build in float64 because the projector checks in data.py are intentionally
    # strict. The tensors used for training are cast to float32 below.
    ds = ManifoldGMM.random_gmm(
        d=int(args.D),
        k=int(args.K),
        n_modes=int(args.M),
        sep=float(args.sep),
        mode_sigma=float(args.mode_sigma),
        off_manifold_sigma=float(args.off_manifold_sigma),
        n_train=int(args.N),
        seed=int(args.geometry_seed),
        dtype=torch.float64,
    )
    train = ds.train_set.detach().cpu().float()

    # The train set is sampled inside ManifoldGMM with geometry_seed + 1. Use a
    # far-away reference seed so the held-out baseline is not identical to train.
    reference_seed = int(args.geometry_seed) + 100003 + int(args.seed)
    reference = ds.sample(int(args.N_test), seed=reference_seed).detach().cpu().float()
    if train.shape == reference.shape and torch.equal(train, reference):
        raise ValueError("train and reference tensors are exactly equal.")
    return ds, train, reference


# =============================================================================
# Small persistence helpers
# =============================================================================


def save_json(path, payload):
    """Write a readable JSON file with stable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path, payload):
    """Append one machine-readable checkpoint metric record."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_csv(path, row):
    """Append one human-friendly checkpoint metric row."""
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in METRIC_FIELDS})


# =============================================================================
# Model, sampling, and metrics
# =============================================================================


def make_optimizer(config, model):
    """Build the optimizer requested by the prototype CLI."""
    if str(config.training.optimizer) == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
            momentum=float(config.training.momentum),
        )
    if str(config.training.optimizer) == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )
    raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def sample_model(model, z_init, config, device):
    """Generate samples from the fixed standard-normal evaluation noise bank."""
    outputs = []
    batch_size = int(config.evaluation.batch_size)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, z_init.shape[0], batch_size):
                # Sampling in batches keeps evaluation memory bounded.
                z = z_init[start : start + batch_size].to(device)
                samples = model.sample(
                    z,
                    n_steps=int(config.sampling.n_steps),
                    method=str(config.sampling.method),
                    device=device,
                )
                outputs.append(samples.detach().cpu().float())
    finally:
        if was_training:
            model.train()
    return torch.cat(outputs, dim=0)


def prefixed(prefix, metrics):
    """Prefix held-out baseline metric names before merging with generated metrics."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def true_gmm_nll(samples, dataset, chunk_size):
    """Compute mean NLL under the exact synthetic ManifoldGMM data density."""
    # This is the vector-data analogue of a quality metric. Unlike FID, it uses
    # the known synthetic distribution directly, so lower NLL means better data
    # likelihood under the ground-truth GMM.
    total_nll = 0.0
    total_count = 0
    log_pi = torch.log(dataset.pi)
    logdet = 2.0 * torch.log(torch.diagonal(dataset.L_tilde, dim1=-2, dim2=-1)).sum(-1)
    const = dataset.d * math.log(2.0 * math.pi)

    with torch.no_grad():
        for start in range(0, samples.shape[0], int(chunk_size)):
            x = samples[start : start + int(chunk_size)].to(
                dataset.device, dataset.dtype
            )
            diff = x.unsqueeze(1) - dataset.mu_tilde.unsqueeze(0)
            L = dataset.L_tilde.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            solved = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
            quad = (diff * solved).sum(-1)
            log_normal = -0.5 * (const + logdet.unsqueeze(0) + quad)
            log_prob = torch.logsumexp(log_pi.unsqueeze(0) + log_normal, dim=1)
            total_nll += float((-log_prob).sum().item())
            total_count += int(x.shape[0])

    return total_nll / max(total_count, 1)


def evaluate_checkpoint(
    model, step, loss, train, reference, z_init, dataset, config, args, run_dir, device
):
    """Sample one checkpoint and compare generated memorization and quality."""
    samples = sample_model(model, z_init, config, device)

    # Generated memorization: compare model samples against the frozen train set.
    generated = compute_memorization_metrics(
        samples,
        train,
        ratio_threshold=float(args.rho),
        chunk_size=int(args.chunk_size),
    )
    generated["gmm_nll"] = true_gmm_nll(samples, dataset, args.chunk_size)
    generated["gmm_nll_per_dim"] = generated["gmm_nll"] / float(dataset.d)
    baseline_count = min(int(args.N_gen), int(reference.shape[0]))

    # Held-out baseline: non-training data should provide the reference level
    # for nearest-neighbor ratios under the same dataset geometry.
    baseline = compute_memorization_metrics(
        reference[:baseline_count],
        train,
        ratio_threshold=float(args.rho),
        chunk_size=int(args.chunk_size),
    )
    baseline["gmm_nll"] = true_gmm_nll(
        reference[:baseline_count], dataset, args.chunk_size
    )
    baseline["gmm_nll_per_dim"] = baseline["gmm_nll"] / float(dataset.d)
    row = {
        "param": str(args.param),
        "step": int(step),
        "loss": None if loss is None else float(loss),
        "sample_count": int(samples.shape[0]),
        "rho": float(args.rho),
        **generated,
        **prefixed("baseline", baseline),
    }
    row["delta_f_mem"] = row["f_mem"] - row["baseline_f_mem"]
    row["delta_mean_nn_ratio"] = row["mean_nn_ratio"] - row["baseline_mean_nn_ratio"]
    row["delta_gmm_nll"] = row["gmm_nll"] - row["baseline_gmm_nll"]
    row["delta_gmm_nll_per_dim"] = (
        row["gmm_nll_per_dim"] - row["baseline_gmm_nll_per_dim"]
    )
    append_csv(run_dir / "metrics" / "checkpoint_metrics.csv", row)
    append_jsonl(run_dir / "metrics" / "checkpoint_metrics.jsonl", row)
    if args.print_eval:
        loss_text = "nan" if loss is None else f"{float(loss):.6g}"
        print(
            "eval "
            f"param={row['param']} step={row['step']} loss={loss_text} "
            f"f_mem={row['f_mem']:.6g} baseline_f_mem={row['baseline_f_mem']:.6g} "
            f"delta_f_mem={row['delta_f_mem']:.6g} "
            f"mean_r={row['mean_nn_ratio']:.6g} "
            f"baseline_mean_r={row['baseline_mean_nn_ratio']:.6g} "
            f"delta_mean_r={row['delta_mean_nn_ratio']:.6g} "
            f"gmm_nll={row['gmm_nll']:.6g} "
            f"baseline_gmm_nll={row['baseline_gmm_nll']:.6g} "
            f"gmm_nll_per_dim={row['gmm_nll_per_dim']:.6g} "
            f"baseline_gmm_nll_per_dim={row['baseline_gmm_nll_per_dim']:.6g}",
            flush=True,
        )
    return row


# =============================================================================
# Training lifecycle
# =============================================================================


def train_one(args):
    """Run the full single-parameterization experiment."""
    validate_args(args)
    device = resolve_device(args.device)

    # Seed model initialization and GDFM's internally sampled training noise.
    torch.manual_seed(int(args.seed))

    run_dir = make_run_dir(args)
    config = build_config(args)
    ds, train, reference = build_dataset(args)

    # This is intentionally standard normal, not sigma-scaled. Sigma controls
    # the data distribution; the GDFM source distribution stays N(0, I).
    z_gen = torch.Generator().manual_seed(int(args.seed) + 2003)
    z_init = torch.randn(int(args.N_gen), int(args.D), generator=z_gen)

    # Save the exact tensors used by this run so checkpoint comparisons can be
    # replayed without rebuilding the dataset.
    torch.save(train, run_dir / "artifacts" / "train_tensor.pt")
    torch.save(reference, run_dir / "artifacts" / "reference_tensor.pt")
    torch.save(z_init, run_dir / "artifacts" / "z_init.pt")
    save_json(run_dir / "artifacts" / "config.json", config_to_dict(config))
    save_json(
        run_dir / "artifacts" / "data_meta.json",
        {
            "dataset": "intr_amb_manifold_gmm",
            "train_shape": list(train.shape),
            "reference_shape": list(reference.shape),
            "param": str(args.param),
            "D": int(args.D),
            "K": int(args.K),
            "M": int(args.M),
            "N": int(args.N),
            "N_test": int(args.N_test),
            "mode_sigma": float(args.mode_sigma),
            "off_manifold_sigma": float(args.off_manifold_sigma),
            "sigma": float(args.mode_sigma),
            "sep": float(args.sep),
            "seed": int(args.seed),
            "geometry_seed": int(args.geometry_seed),
            "optimizer": str(args.optimizer),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "momentum": float(args.momentum),
            "source_noise": "standard_normal",
        },
    )

    model = GDFM(config).to(device)
    optimizer = make_optimizer(config, model)
    ckpts = set(checkpoint_steps(args))
    history = []
    last_loss = None

    # Step 0 gives the untrained baseline under the same sampler and metrics.
    save_checkpoint(run_dir, 0, model, optimizer)
    metric_rows = [
        evaluate_checkpoint(
            model, 0, None, train, reference, z_init, ds, config, args, run_dir, device
        )
    ]

    # This generator controls minibatch index draws only. GDFM still samples its
    # own training time t and source noise eps inside compute_loss(...).
    batch_gen = torch.Generator().manual_seed(int(args.seed) + 17)
    progress = range(1, int(args.tau) + 1)
    if not args.no_progress:
        from tqdm.auto import tqdm

        progress = tqdm(progress, desc=f"train:{args.param}", unit="step")

    for step in progress:
        # Draw with replacement from the fixed training tensor. This keeps the
        # training loop short and explicit for the prototype.
        idx = torch.randint(
            0,
            train.shape[0],
            (min(int(args.B), train.shape[0]),),
            generator=batch_gen,
        )
        batch = train[idx].to(device)

        # One FM optimizer step: construct z_t and targets inside GDFM, then
        # update the MLP for the selected parameterization.
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        # Keep a sparse loss curve for quick diagnostics.
        if step == 1 or step % int(args.tau_log) == 0 or step == int(args.tau):
            history.append({"step": int(step), "loss": last_loss})
            save_json(run_dir / "metrics" / "train_history.json", history)
            if not args.no_progress:
                progress.set_postfix(loss=f"{last_loss:.4f}", refresh=False)

        # Checkpoint metrics are the main output: generation, f_mem, and NN
        # ratio all use the same fixed z_init across checkpoints.
        if int(step) in ckpts:
            save_checkpoint(run_dir, step, model, optimizer)
            metric_rows.append(
                evaluate_checkpoint(
                    model,
                    step,
                    last_loss,
                    train,
                    reference,
                    z_init,
                    ds,
                    config,
                    args,
                    run_dir,
                    device,
                )
            )

    save_json(run_dir / "metrics" / "train_history.json", history)
    save_figures(metric_rows, run_dir)
    save_final_2d_visualizations(model, train, z_init, config, args, run_dir, device)
    save_json(
        run_dir / "artifacts" / "run_summary.json",
        {
            "run_dir": str(run_dir.resolve()),
            "param": str(args.param),
            "tau": int(args.tau),
            "checkpoint_steps": sorted(int(step) for step in ckpts),
            "final_loss": last_loss,
            "optimizer": str(args.optimizer),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "momentum": float(args.momentum),
            "metrics_path": str(
                (run_dir / "metrics" / "checkpoint_metrics.csv").resolve()
            ),
            "figures_dir": str((run_dir / "figures").resolve()),
        },
    )
    return run_dir


# =============================================================================
# 2D sampling visualization
# =============================================================================


def record_sampling_trajectory(model, z_init, config, device, max_frames=80):
    """Record a small set of intermediate 2D states from the active sampler."""
    n_steps = int(config.sampling.n_steps)
    method = str(config.sampling.method)
    frame_stride = max(1, n_steps // int(max_frames))
    batch_size = int(z_init.shape[0])
    was_training = model.training
    model.eval()

    with torch.no_grad():
        z = z_init.to(device)
        ts = torch.linspace(model.t_min, model.t_max, n_steps + 1, device=device)
        frames = [(0, float(ts[0].item()), z.detach().cpu().float())]

        try:
            for index in range(len(ts) - 1):
                dt = ts[index + 1] - ts[index]
                if method == "euler":
                    velocity = model._velocity(z, ts[index].item(), batch_size)
                    z = z + dt * velocity
                elif method == "heun":
                    velocity_1 = model._velocity(z, ts[index].item(), batch_size)
                    z_tilde = z + dt * velocity_1
                    velocity_2 = model._velocity(
                        z_tilde, ts[index + 1].item(), batch_size
                    )
                    z = z + 0.5 * dt * (velocity_1 + velocity_2)
                else:
                    raise ValueError(f"Unknown sampling method: {method}")

                step = index + 1
                if step % frame_stride == 0 or step == n_steps:
                    frames.append(
                        (step, float(ts[step].item()), z.detach().cpu().float())
                    )
        finally:
            if was_training:
                model.train()

    return frames


def axis_limits_2d(*tensors):
    """Build shared 2D axis limits so source, train, and generated are comparable."""
    points = torch.cat(
        [tensor.detach().cpu().float()[:, :2] for tensor in tensors], dim=0
    )
    lo = points.min(dim=0).values
    hi = points.max(dim=0).values
    center = 0.5 * (lo + hi)
    radius = 0.55 * float((hi - lo).max().item())
    radius = max(radius, 1.0e-3)
    return (
        float(center[0] - radius),
        float(center[0] + radius),
        float(center[1] - radius),
        float(center[1] + radius),
    )


def save_final_2d_visualizations(model, train, z_init, config, args, run_dir, device):
    """Save final source/generated/train scatter and an MP4 sampling trajectory."""
    if int(args.D) != 2:
        return

    os.environ.setdefault("XDG_CACHE_HOME", str((run_dir / "logs" / "cache").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((run_dir / "logs" / "mpl").resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    generated = sample_model(model, z_init, config, device)
    x_min, x_max, y_min, y_max = axis_limits_2d(z_init, generated, train)

    # Final scatter: source noise, final generated samples, and frozen train set.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        train[:, 0],
        train[:, 1],
        s=18,
        c="black",
        alpha=0.30,
        label="train",
        linewidths=0,
    )
    ax.scatter(
        z_init[:, 0],
        z_init[:, 1],
        s=14,
        c="#4c78a8",
        alpha=0.12,
        label="source z_init",
        linewidths=0,
    )
    ax.scatter(
        generated[:, 0],
        generated[:, 1],
        s=14,
        c="#f58518",
        alpha=0.65,
        label="generated",
        linewidths=0,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"final 2D samples: param={args.param}, sampler={config.sampling.method}"
    )
    ax.grid(alpha=0.20)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "final_2d_source_generated_train.png", dpi=180)
    plt.close(fig)

    # Sampling movie: fixed particles move from source noise toward generated data.
    if not animation.writers.is_available("ffmpeg"):
        print(
            "2D video skipped: matplotlib ffmpeg writer is not available.", flush=True
        )
        return

    frames = record_sampling_trajectory(model, z_init, config, device)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        train[:, 0],
        train[:, 1],
        s=18,
        c="black",
        alpha=0.22,
        label="train",
        linewidths=0,
    )
    particles = ax.scatter(
        frames[0][2][:, 0],
        frames[0][2][:, 1],
        s=14,
        c="#f58518",
        alpha=0.70,
        label="samples",
        linewidths=0,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.legend(loc="best")

    def update(frame):
        step, t_value, points = frame
        particles.set_offsets(points[:, :2])
        ax.set_title(
            f"sampling: param={args.param}, sampler={config.sampling.method}, "
            f"step={step}/{config.sampling.n_steps}, t={t_value:.3f}"
        )
        return (particles,)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=80,
        blit=False,
        repeat=False,
    )
    try:
        writer = animation.FFMpegWriter(fps=12, bitrate=1800)
        movie.save(run_dir / "figures" / "final_2d_sampling_process.mp4", writer=writer)
    except Exception as exc:
        print(f"2D video skipped: failed to write MP4 ({exc}).", flush=True)
    finally:
        plt.close(fig)


# =============================================================================
# Figures
# =============================================================================


def save_figures(rows, run_dir):
    """Save the two comparison curves used for quick visual inspection."""
    # Some cluster/login environments have unwritable default font caches.
    os.environ.setdefault("XDG_CACHE_HOME", str((run_dir / "logs" / "cache").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((run_dir / "logs" / "mpl").resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    baseline_f_mem = [float(row["baseline_f_mem"]) for row in rows]
    baseline_mean_r = [float(row["baseline_mean_nn_ratio"]) for row in rows]
    baseline_gmm_nll = [float(row["baseline_gmm_nll"]) for row in rows]

    # Dashed held-out curves are the non-memorized reference level.
    figure_specs = [
        ("f_mem_over_steps.png", "f_mem", "f_mem", baseline_f_mem),
        (
            "mean_nn_ratio_over_steps.png",
            "mean_nn_ratio",
            "mean d1/d2 (lower = closer to train)",
            baseline_mean_r,
        ),
        (
            "gmm_nll_over_steps.png",
            "gmm_nll",
            "true GMM NLL (lower = better quality)",
            baseline_gmm_nll,
        ),
    ]
    for filename, key, ylabel, baseline in figure_specs:
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, values, marker="o", linewidth=1.8, label="generated")
        ax.plot(steps, baseline, linestyle="--", linewidth=1.4, label="held-out")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / filename, dpi=160)
        plt.close(fig)

    # Delta curves answer the key question more directly when the held-out
    # baseline is already close to the train set under the raw NN-ratio metric.
    delta_specs = [
        (
            "delta_f_mem_over_steps.png",
            "delta_f_mem",
            "generated f_mem - held-out f_mem",
        ),
        (
            "delta_mean_nn_ratio_over_steps.png",
            "delta_mean_nn_ratio",
            "generated mean d1/d2 - held-out mean d1/d2",
        ),
        (
            "delta_gmm_nll_over_steps.png",
            "delta_gmm_nll",
            "generated GMM NLL - held-out GMM NLL",
        ),
    ]
    for filename, key, ylabel in delta_specs:
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.plot(steps, values, marker="o", linewidth=1.8)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / filename, dpi=160)
        plt.close(fig)


def main(argv=None):
    """CLI entrypoint."""
    args = parse_args(argv)
    run_dir = train_one(args)
    print(f"run_dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
