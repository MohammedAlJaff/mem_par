from pathlib import Path

import torch

from gdfm import GDFM
from mp_artifacts import (
    create_run_layout,
    save_checkpoint,
    save_config_snapshot,
    save_json,
    save_tensor,
)
from mp_data import build_data_bundle


def _build_optimizer(config, model):
    name = str(config.training.optimizer).lower()

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
            momentum=float(config.training.momentum),
        )

    raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def _sample_batch(train_tensor, batch_size, generator):
    indices = torch.randint(
        0, train_tensor.shape[0], (batch_size,), generator=generator
    )
    return train_tensor[indices]


def _make_z_init(config, train_tensor):
    sample_shape = tuple(train_tensor.shape[1:])
    return torch.randn(int(config.evaluation.n_samples), *sample_shape)


def train_run(config, study_dir, device="cpu"):
    study_dir = Path(study_dir)
    run_dir = create_run_layout(study_dir, config)
    data_bundle = build_data_bundle(config)

    train_tensor = data_bundle["train"].cpu()
    reference_tensor = data_bundle["reference"].cpu()
    z_init = _make_z_init(config, train_tensor)

    save_config_snapshot(run_dir, config)
    save_tensor(run_dir / "artifacts" / "train_tensor.pt", train_tensor)
    save_tensor(run_dir / "artifacts" / "reference_tensor.pt", reference_tensor)
    save_tensor(run_dir / "artifacts" / "z_init.pt", z_init)

    if data_bundle.get("train_labels") is not None:
        save_tensor(run_dir / "artifacts" / "train_labels.pt", data_bundle["train_labels"])
    if data_bundle.get("reference_labels") is not None:
        save_tensor(
            run_dir / "artifacts" / "reference_labels.pt",
            data_bundle["reference_labels"],
        )
    if data_bundle.get("train_indices") is not None:
        save_tensor(run_dir / "artifacts" / "train_indices.pt", data_bundle["train_indices"])
    if data_bundle.get("reference_indices") is not None:
        save_tensor(
            run_dir / "artifacts" / "reference_indices.pt",
            data_bundle["reference_indices"],
        )
    if data_bundle.get("geometry") is not None:
        save_tensor(run_dir / "artifacts" / "geometry.pt", data_bundle["geometry"])

    save_json(
        run_dir / "artifacts" / "data_meta.json",
        {
            "display_shape": data_bundle.get("display_shape"),
            "train_shape": list(train_tensor.shape),
            "reference_shape": list(reference_tensor.shape),
        },
    )

    model = GDFM(config).to(device)
    optimizer = _build_optimizer(config, model)
    generator = torch.Generator().manual_seed(int(config.training.seed))

    history = []
    save_checkpoint(run_dir, 0, model, optimizer)

    for step in range(1, int(config.training.n_steps) + 1):
        batch = _sample_batch(
            train_tensor,
            min(int(config.training.batch_size), train_tensor.shape[0]),
            generator,
        ).to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        if step == 1 or step % int(config.training.log_every) == 0 or step == int(config.training.n_steps):
            history.append({"step": int(step), "loss": float(loss.item())})

        if step % int(config.training.ckpt_every) == 0 or step == int(config.training.n_steps):
            save_checkpoint(run_dir, step, model, optimizer)

    save_json(run_dir / "metrics" / "train_history.json", history)
    save_json(
        run_dir / "artifacts" / "run_summary.json",
        {
            "study": config.study.name,
            "run_dir": str(run_dir),
            "n_steps": int(config.training.n_steps),
            "n_train": int(config.data.n_train),
            "n_eval": int(config.data.n_eval),
            "optimizer": str(config.training.optimizer),
        },
    )
    return run_dir
