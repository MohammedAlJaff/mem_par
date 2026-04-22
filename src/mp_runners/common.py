"""
Small helpers shared by canonical runner modules.

Keep this module limited to direct runner utilities: parameter counts,
batch-cap handling, JSON printing, and standard artifact path summaries.
"""

import json
from pathlib import Path


PARAMETERIZATION_CHOICES = ("x", "eps", "v")


def print_json(payload):
    """Print one JSON-friendly object with stable indentation."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def as_int_tuple(text):
    """Parse CLI tuple text such as ``1,2,4`` into integers."""
    if isinstance(text, tuple):
        return text
    if isinstance(text, list):
        return tuple(int(item) for item in text)
    return tuple(int(item.strip()) for item in str(text).split(",") if item.strip())


def count_parameters(model):
    """Count total and trainable parameters for quick inspection."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def resolve_batch_cap(knobs):
    """Resolve the public batch-cap setting for runner recipes."""
    batch_cap = knobs.get("batch_cap")
    if batch_cap is None:
        raise ValueError("Runner settings must define batch_cap.")
    batch_cap = int(batch_cap)
    if batch_cap < 1:
        raise ValueError(f"batch_cap must be >= 1, got {batch_cap}")
    return batch_cap


def resolve_batch_size(n_train, batch_cap):
    """Return the effective mini-batch size for a selected dataset."""
    if int(n_train) < 1:
        raise ValueError("n_train must be at least 1.")
    batch_cap = int(batch_cap)
    if batch_cap < 1:
        raise ValueError(f"batch_cap must be >= 1, got {batch_cap}")
    return min(int(n_train), batch_cap)


def artifact_paths(attempt_dir):
    """
    Return the standard files to inspect under one attempt directory.

    Runner printouts use these paths so a user can find checkpoints, metrics,
    figures, logs, and summaries without knowing the full experiment layout.
    """
    attempt_dir = Path(attempt_dir)
    paths = {
        "attempt_dir": attempt_dir,
        "status": attempt_dir / "status.json",
        "config": attempt_dir / "artifacts" / "config.json",
        "data_meta": attempt_dir / "artifacts" / "data_meta.json",
        "run_summary": attempt_dir / "artifacts" / "run_summary.json",
        "train_history": attempt_dir / "metrics" / "train_history.json",
        "checkpoint_metrics": attempt_dir / "metrics" / "checkpoint_metrics.json",
        "loss_eval": attempt_dir / "metrics" / "loss_eval.json",
        "checkpoint_sample_previews": (
            attempt_dir / "metrics" / "checkpoint_sample_previews.json"
        ),
        "final_samples": attempt_dir / "metrics" / "final_samples.pt",
        "figures": attempt_dir / "figures",
        "checkpoints": attempt_dir / "checkpoints",
        "logs": attempt_dir / "logs",
    }
    return {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in paths.items()
    }


def latest_completed_attempt(study_dir):
    """Find the newest completed attempt recorded in a study manifest."""
    study_dir = Path(study_dir)
    manifest_path = study_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        return None

    # Walk the canonical experiment -> run -> seed -> attempt manifests instead
    # of guessing from filesystem mtimes. This mirrors how study code locates
    # completed attempts after cluster runs.
    manifest = json.loads(manifest_path.read_text())
    candidates = []
    for run_entry in manifest.get("runs", {}).values():
        run_manifest_name = run_entry.get("manifest_path")
        if not run_manifest_name:
            continue
        run_manifest_path = study_dir / run_manifest_name
        if not run_manifest_path.is_file():
            continue
        run_manifest = json.loads(run_manifest_path.read_text())
        for seed_entry in run_manifest.get("seeds", {}).values():
            seed_manifest_name = seed_entry.get("manifest_path")
            if not seed_manifest_name:
                continue
            seed_manifest_path = study_dir / seed_manifest_name
            if not seed_manifest_path.is_file():
                continue
            seed_manifest = json.loads(seed_manifest_path.read_text())
            latest_id = seed_manifest.get("latest_completed_attempt")
            if latest_id is None:
                continue
            attempt_entry = seed_manifest.get("attempts", {}).get(latest_id, {})
            attempt_path = study_dir / attempt_entry.get("path", "")
            candidates.append(
                (
                    attempt_entry.get("finished_at")
                    or attempt_entry.get("updated_at")
                    or attempt_entry.get("created_at")
                    or "",
                    attempt_path,
                )
            )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


if __name__ == "__main__":
    print("mp_runners.common example")
    print("parameterization choices:", ", ".join(PARAMETERIZATION_CHOICES))
