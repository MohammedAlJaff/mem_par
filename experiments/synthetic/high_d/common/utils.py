"""
Utilities: deterministic seeding, config I/O, logging helpers.

Design principle: every random operation in the codebase should be
traceable to a seed. This is non-negotiable for a paper about phase
boundaries — if results aren't reproducible, the boundary location
is meaningless.
"""

import os
import json
import yaml
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any


# =============================================================================
# Seeding
# =============================================================================
# WHY: We need *exact* reproducibility within a (config, seed) pair.
# torch, numpy, and python's random module each have independent PRNGs.
# We must seed all three. CUDA adds another layer — cuDNN has nondeterministic
# algorithms that are faster but not reproducible. We disable those.

def seed_everything(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Call this ONCE at the start of each run, before any data generation
    or model initialization.

    Note on CUDA determinism: setting torch.use_deterministic_algorithms(True)
    would be ideal but breaks some operations (e.g., upsample_nearest2d backward
    on CUDA). We use the softer cudnn settings instead, which cover most cases.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic. This has a ~10-15% speed cost.
    # If you're debugging and need speed, you can disable these,
    # but NEVER for final experimental runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Config I/O
# =============================================================================
# Configs use YAML for human readability. We support a simple _base_ inheritance
# mechanism: if a config has a _base_ key, we load that file first and merge.

def load_config(path: str) -> dict:
    """
    Load a YAML config file with optional base inheritance.

    If the config contains a '_base_' key, that base config is loaded first,
    then the current config's values override it. Only one level of inheritance
    is supported (no recursive bases).

    Example:
        # experiment.yaml
        _base_: base.yaml
        data:
          n: [100, 200]    # overrides base
    """
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    # Handle base config inheritance
    if "_base_" in config:
        base_path = path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        config = deep_merge(base_config, config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.

    - Dict values are merged recursively
    - All other values in override replace base
    - Keys in base but not in override are preserved

    This is the standard "deep update" pattern. It means you only need
    to specify the keys you want to change in the child config.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict, path: str) -> None:
    """Save config as JSON (not YAML — JSON is safer for exact reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)


# =============================================================================
# Run directory management
# =============================================================================

def make_run_dir(base_dir: str, config: dict) -> Path:
    """
    Create a unique run directory from config parameters.

    Naming: {experiment}/{pred}_{loss}_d{d}_n{n}_seed{s}
    This makes it easy to grep for specific conditions.
    """
    exp = config.get("experiment", "unknown")
    pred = config.get("pred_space", "x")
    loss = config.get("loss_space", "v")
    seed = config.get("seed", 0)

    # Build name from whatever axes are present
    parts = [f"pred{pred}", f"loss{loss}"]
    if "d" in config:
        parts.append(f"d{config['d']}")
    if "n" in config:
        parts.append(f"n{config['n']}")
    parts.append(f"seed{seed}")

    run_name = "_".join(parts)
    run_dir = Path(base_dir) / exp / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    return run_dir


# =============================================================================
# Logging helpers
# =============================================================================

def log_message(msg: str, run_dir: Path = None) -> None:
    """Print timestamped message and optionally append to run log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if run_dir is not None:
        with open(run_dir / "run.log", "a") as f:
            f.write(line + "\n")


def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics dict as JSON. Handles numpy/torch types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj

    serializable = json.loads(json.dumps(metrics, default=convert))
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_metrics(path: str) -> dict:
    """Load metrics JSON."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Dataset persistence
# =============================================================================
# WHY: In memorization research the training set is the ground truth.
# Every NN comparison (f_mem, d1, d2, r) must be made against the EXACT
# tensor the model trained on — not a re-generated approximation.
#
# For synthetic data: seeds + params theoretically reproduce the tensor,
# but code changes, numpy version changes, or a typo in a param can silently
# yield a different set. Saving the tensor removes all ambiguity.
#
# For image data: subsample_deterministic is seeded, but filesystem sort order
# can differ across machines/OS. Saving the indices is the only safe record.
#
# RULE: call save_dataset_artifact() once, right after build_dataset(),
# before training starts. Load train_data.pt in evaluation instead of
# regenerating the dataset.


def save_dataset_artifact(
    dataset,
    run_dir: Path,
    dataset_params: dict,
) -> Path:
    """
    Persist the training dataset alongside the run directory.

    Writes three files:
        run_dir/train_data.pt       — (n, *) tensor, the exact training set
        run_dir/train_indices.pt    — (n,) long tensor of indices into the
                                      parent dataset (image subsets only;
                                      for synthetic data this is 0..n-1)
        run_dir/dataset_info.json   — all params that produced this data

    After saving, evaluations must load train_data.pt instead of rebuilding
    the dataset. This guarantees the comparison is against what was trained on.

    Args:
        dataset:         a torch Dataset (GMMSubspace, TorusEmbedding,
                         CelebA32Grey, ImageFolderSubset, or Subset thereof)
        run_dir:         Path to the run directory (must exist)
        dataset_params:  dict of params used to build the dataset
                         (type, n, seed, D, d, etc.) — saved verbatim to JSON

    Returns:
        Path to train_data.pt (useful for logging)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect the full data tensor ----
    # Use a DataLoader for memory efficiency; avoids one giant allocation.
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    chunks = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]   # (img, label) → img
        chunks.append(batch.cpu())
    train_data = torch.cat(chunks, dim=0)   # (n, *)

    data_path = run_dir / "train_data.pt"
    torch.save(train_data, data_path)

    # ---- Save indices ----
    # For torch Subset (image datasets): use the stored .indices list.
    # For synthetic datasets (all data is self-generated): indices are 0..n-1.
    indices = _extract_indices(dataset, len(train_data))
    torch.save(torch.tensor(indices, dtype=torch.long), run_dir / "train_indices.pt")

    # ---- Save metadata ----
    info = {
        "n_saved": len(train_data),
        "shape": list(train_data.shape),
        "dtype": str(train_data.dtype),
        **dataset_params,
    }
    with open(run_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    return data_path


def load_train_data(run_dir: Path) -> torch.Tensor:
    """
    Load the saved training tensor for a run.

    Always use this in evaluation scripts instead of rebuilding the dataset.
    The tensor is exactly what the model trained on.

    Args:
        run_dir: Path to the run directory

    Returns:
        (n, *) tensor on CPU
    """
    path = Path(run_dir) / "train_data.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"train_data.pt not found in {run_dir}. "
            f"Was save_dataset_artifact() called for this run?"
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def _extract_indices(dataset, n: int) -> list:
    """
    Extract integer indices from a dataset.

    For torch.utils.data.Subset: returns the stored .indices.
    For wrapped datasets (CelebA32Grey, ImageFolderSubset): walk the .dataset
    attribute chain until we find a Subset or give up.
    For synthetic datasets (GMMSubspace, TorusEmbedding): indices are 0..n-1
    because all samples are self-generated with no parent dataset.
    """
    from torch.utils.data import Subset

    # Walk the .dataset chain looking for a Subset
    obj = dataset
    for _ in range(5):   # max depth to avoid infinite loops
        if isinstance(obj, Subset):
            return list(obj.indices)
        if hasattr(obj, "dataset"):
            obj = obj.dataset
        else:
            break

    # No Subset found — synthetic dataset, indices are positional
    return list(range(n))


# =============================================================================
# Self-test  (run with: python -m src.utils  from project root)
# =============================================================================

if __name__ == "__main__":
    import tempfile
    print("=== experiments.synthetic.high_d.common.utils self-test ===\n")

    # --- seed_everything ---
    seed_everything(42)
    a = np.random.randn(3)
    seed_everything(42)
    b = np.random.randn(3)
    assert np.allclose(a, b), "seed_everything is not deterministic"
    print(f"[PASS] seed_everything(42) → reproducible numpy draws: {a.round(4)}")

    import torch as _torch
    seed_everything(7)
    ta = _torch.randn(3)
    seed_everything(7)
    tb = _torch.randn(3)
    assert _torch.allclose(ta, tb), "seed_everything torch not deterministic"
    print(f"[PASS] seed_everything(7) → reproducible torch draws: {ta.numpy().round(4)}")

    # --- load_config / deep_merge ---
    with tempfile.TemporaryDirectory() as td:
        base_path = Path(td) / "base.yaml"
        child_path = Path(td) / "child.yaml"
        base_path.write_text("a: 1\nb:\n  x: 10\n  y: 20\n")
        child_path.write_text(f"_base_: base.yaml\nb:\n  y: 99\nc: 3\n")
        cfg = load_config(str(child_path))
    assert cfg["a"] == 1,  "base key not inherited"
    assert cfg["b"]["x"] == 10, "nested base key not inherited"
    assert cfg["b"]["y"] == 99, "child override not applied"
    assert cfg["c"] == 3,  "child-only key missing"
    print(f"[PASS] load_config with _base_ inheritance → {cfg}")

    # --- save_config / save_metrics / load_metrics ---
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "sub" / "config.json"
        save_config({"lr": 1e-3, "n": 500}, str(cfg_path))
        assert cfg_path.exists(), "config file not created"
        print(f"[PASS] save_config created {cfg_path.name}")

        metrics = {"loss": np.float32(0.123), "step": np.int64(42), "arr": np.array([1.0, 2.0])}
        m_path = Path(td) / "metrics.json"
        save_metrics(metrics, str(m_path))
        loaded = load_metrics(str(m_path))
        assert abs(loaded["loss"] - 0.123) < 1e-4
        assert loaded["step"] == 42
        assert loaded["arr"] == [1.0, 2.0]
        print(f"[PASS] save_metrics / load_metrics roundtrip: {loaded}")

    # --- make_run_dir ---
    with tempfile.TemporaryDirectory() as td:
        rc = {"experiment": "test", "pred_space": "x", "loss_space": "v", "d": 4, "n": 100, "seed": 0}
        run_dir = make_run_dir(td, rc)
        assert (run_dir / "checkpoints").exists()
        assert (run_dir / "samples").exists()
        assert (run_dir / "figures").exists()
        assert "predx_lossv_d4_n100_seed0" in str(run_dir)
        print(f"[PASS] make_run_dir → {run_dir.relative_to(td)}")

    # --- log_message ---
    with tempfile.TemporaryDirectory() as td:
        log_message("hello from test", Path(td))
        log_path = Path(td) / "run.log"
        assert log_path.exists() and "hello from test" in log_path.read_text()
        print("[PASS] log_message writes to run.log")

    # --- save_dataset_artifact / load_train_data ---
    import torch as _torch
    from torch.utils.data import TensorDataset, Subset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Synthetic-style dataset (no parent Subset)
        raw = _torch.randn(50, 8)
        synth_ds = TensorDataset(raw)
        params = {"type": "gmm_subspace", "n": 50, "D": 8, "d": 4, "seed": 0}
        data_path = save_dataset_artifact(synth_ds, td / "run_synth", params)
        loaded = load_train_data(td / "run_synth")
        assert loaded.shape == (50, 8), f"Expected (50,8), got {loaded.shape}"
        assert _torch.allclose(loaded, raw), "Loaded data doesn't match original"
        idx = _torch.load(td / "run_synth" / "train_indices.pt", weights_only=True)
        assert idx.tolist() == list(range(50)), "Synthetic indices should be 0..n-1"
        print(f"[PASS] save_dataset_artifact (synthetic): shape={tuple(loaded.shape)}")

        # Image-style dataset with a Subset (indices should be saved)
        full_ds = TensorDataset(_torch.randn(100, 3, 8, 8))
        chosen_idx = [2, 7, 15, 42, 99]
        subset_ds = Subset(full_ds, chosen_idx)
        data_path2 = save_dataset_artifact(subset_ds, td / "run_image",
                                            {"type": "image_folder", "n": 5, "seed": 1})
        loaded2 = load_train_data(td / "run_image")
        assert loaded2.shape == (5, 3, 8, 8)
        idx2 = _torch.load(td / "run_image" / "train_indices.pt", weights_only=True)
        assert idx2.tolist() == chosen_idx, f"Expected {chosen_idx}, got {idx2.tolist()}"
        print(f"[PASS] save_dataset_artifact (image subset): indices={idx2.tolist()}")

        # load_train_data raises if file missing
        try:
            load_train_data(td / "nonexistent_run")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("[PASS] load_train_data raises FileNotFoundError when file missing")

    print("\n=== All utils tests passed ===")
