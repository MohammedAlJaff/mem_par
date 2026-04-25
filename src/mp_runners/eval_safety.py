"""
Shared saved-run validation for checkpoint replay.
"""

from pathlib import Path

import torch

from mp_artifacts import load_json


def _indices_to_set(path):
    """Load saved source indices as a Python set."""
    return {
        int(item)
        for item in torch.load(path, map_location="cpu").flatten().tolist()
    }


def validate_saved_train_reference(run_dir, train_tensor, reference_tensor):
    """
    Validate saved train/reference tensors and same-source index metadata.
    """
    run_dir = Path(run_dir)
    if train_tensor.shape == reference_tensor.shape and torch.equal(
        train_tensor,
        reference_tensor,
    ):
        raise ValueError(
            "Data leakage guard failed during evaluation: train and reference "
            "tensors are exactly equal."
        )

    data_meta_path = run_dir / "artifacts" / "data_meta.json"
    if not data_meta_path.exists():
        return

    data_meta = load_json(data_meta_path)
    train_source_id = data_meta.get("train_source_id")
    reference_source_id = data_meta.get("reference_source_id")
    if train_source_id is None or reference_source_id is None:
        return
    if str(train_source_id) != str(reference_source_id):
        return

    train_indices_path = run_dir / "artifacts" / "train_indices.pt"
    reference_indices_path = run_dir / "artifacts" / "reference_indices.pt"
    if not train_indices_path.exists() or not reference_indices_path.exists():
        raise ValueError(
            "Data leakage guard failed during evaluation: train/reference share "
            f"source {train_source_id!r}, but saved source indices are missing."
        )

    overlap = _indices_to_set(train_indices_path).intersection(
        _indices_to_set(reference_indices_path)
    )
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            "Data leakage guard failed during evaluation: saved train/reference "
            f"indices overlap for source {train_source_id!r}. "
            f"overlap_count={len(overlap)}, examples={examples}"
        )

    validation = data_meta.get("data_validation") or {}
    if not bool(validation.get("same_source_allowed", False)):
        raise ValueError(
            "Data leakage guard failed during evaluation: train/reference share "
            f"source {train_source_id!r}. This run was not marked as an allowed "
            "same-source disjoint holdout."
        )


def eval_z_init_bank(config, z_init):
    """
    Return the saved eval bank slice requested by config, or fail clearly.
    """
    requested = int(
        getattr(
            getattr(config, "evaluation", None),
            "n_samples",
            int(z_init.shape[0]),
        )
    )
    if requested < 1:
        raise ValueError(f"evaluation.n_samples must be >= 1, got {requested}")
    available = int(z_init.shape[0])
    if available < requested:
        raise ValueError(
            "Saved z_init bank is smaller than requested checkpoint eval samples: "
            f"available={available}, requested={requested}."
        )
    return z_init[:requested]
