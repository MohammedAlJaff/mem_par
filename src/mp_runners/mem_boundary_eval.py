"""Checkpoint-quality evals for mem-boundary image runs.

The all-DONE mode treats training attempts as read-only inputs, discovers a
stable manifest from a job-set directory, and writes every eval artifact under a
separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
from torchvision import datasets

from gdfm import GDFM
from mp_artifacts import list_checkpoint_paths, load_config_snapshot
from mp_data.celeba32 import _build_transform, _stack_subset
from mp_data.common import seeded_indices
from mp_metrics.loss_eval import loss_eval_times, time_key
from mp_metrics.metrics_images import (
    DEFAULT_FID_FEATURE_EXTRACTOR,
    RHO_SWEEP,
    _TorchFidelityFeatureEncoder,
    _encode_with_torch_fidelity_encoder,
    collect_image32_metric_record,
)
from mp_runtime import autocast_context, configure_torch_runtime


PARAM_SPACES: dict[str, tuple[str, str]] = {
    "x_x": ("x", "x"),
    "eps_eps": ("eps", "eps"),
    "v_v": ("v", "v"),
    "x_v": ("x", "v"),
}
PARAM_ORDER = {param: index for index, param in enumerate(PARAM_SPACES)}
RUN_RE = re.compile(
    r"celeba32_N(?P<n_train>\d+)_B(?P<batch>\d+)_(?:(?P<model_label>[A-Za-z0-9]+_m\d+)_)?"
    r"(?P<param>x_x|eps_eps|v_v|x_v)_"
    r"steps(?P<train_steps>\d+)_s(?P<seed>\d+)-"
)

DEFAULT_SOURCE_ROOT = Path("outputs/job_sets/mem_gen_boundary_20260427")
DEFAULT_OUT_DIR = Path("outputs/evals/mem_boundary_20260427_all_done_quality_20260501")
DEFAULT_PARAMS = "x_x,eps_eps,v_v,x_v"
DEFAULT_SAMPLER = "heun"
DEFAULT_SAMPLING_STEPS = 50
DEFAULT_HELDOUT_SPLIT = "valid"
FID_IMAGE_RANGE = "-1_1"


@dataclass(frozen=True)
class ManifestRow:
    manifest_index: int
    n_train: int
    seed: int
    param: str
    batch_size: int
    train_steps: int
    run_id: str
    attempt_id: str
    attempt_dir: str
    checkpoint_count: int
    final_step: int
    sampler: str
    sampling_steps: int


@dataclass(frozen=True)
class EvalKey:
    n_train: int
    seed: int
    param: str
    step: int
    sample_count: int
    fid_reference_count: int
    heldout_split: str
    sampler: str
    sampling_steps: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "EvalKey | None":
        try:
            return cls(
                n_train=int(row["n_train"]),
                seed=int(row["seed"]),
                param=str(row["param"]),
                step=int(row["step"]),
                sample_count=int(row["sample_count"]),
                fid_reference_count=int(row["fid_reference_count"]),
                heldout_split=str(row["heldout_split"]),
                sampler=str(row["sampler"]),
                sampling_steps=int(row["sampling_steps"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class FidReference:
    mu: np.ndarray
    sigma: np.ndarray
    count: int
    feature_extractor: str
    feature_layer: str
    cache_path: Path
    from_cache: bool


def parse_params(raw: str) -> list[str]:
    params = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [param for param in params if param not in PARAM_SPACES]
    if unknown:
        known = ", ".join(PARAM_SPACES)
        raise ValueError(f"Unknown param(s) {unknown}; expected one of: {known}")
    return params


def parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step_"):
        raise ValueError(f"Unexpected checkpoint filename: {path}")
    return int(stem.removeprefix("step_"))


def manifest_sort_key(row: ManifestRow) -> tuple[int, int, int, str]:
    return (int(row.n_train), int(row.seed), PARAM_ORDER[row.param], row.run_id)


def discover_done_manifest(source_root: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    attempts = sorted(path for path in source_root.glob("runs/*/seeds/seed_*/attempts/DONE_*") if path.is_dir())
    for attempt in attempts:
        run_id = attempt.parents[3].name
        match = RUN_RE.search(run_id)
        if match is None:
            continue
        config = load_config_snapshot(attempt)
        checkpoints = sorted(list_checkpoint_paths(attempt), key=checkpoint_step)
        if not checkpoints:
            final_step = -1
        else:
            final_step = checkpoint_step(checkpoints[-1])
        rows.append(
            ManifestRow(
                manifest_index=-1,
                n_train=int(match.group("n_train")),
                seed=int(match.group("seed")),
                param=str(match.group("param")),
                batch_size=int(match.group("batch")),
                train_steps=int(match.group("train_steps")),
                run_id=run_id,
                attempt_id=attempt.name,
                attempt_dir=str(attempt),
                checkpoint_count=len(checkpoints),
                final_step=final_step,
                sampler=str(getattr(config.sampling, "method", DEFAULT_SAMPLER)),
                sampling_steps=int(getattr(config.sampling, "n_steps", DEFAULT_SAMPLING_STEPS)),
            )
        )
    rows.sort(key=manifest_sort_key)
    return [ManifestRow(**{**asdict(row), "manifest_index": index}) for index, row in enumerate(rows)]


def load_manifest_csv(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                ManifestRow(
                    manifest_index=int(raw["manifest_index"]),
                    n_train=int(raw["n_train"]),
                    seed=int(raw["seed"]),
                    param=str(raw["param"]),
                    batch_size=int(raw["batch_size"]),
                    train_steps=int(raw["train_steps"]),
                    run_id=str(raw["run_id"]),
                    attempt_id=str(raw["attempt_id"]),
                    attempt_dir=str(raw["attempt_dir"]),
                    checkpoint_count=int(raw["checkpoint_count"]),
                    final_step=int(raw["final_step"]),
                    sampler=str(raw["sampler"]),
                    sampling_steps=int(raw["sampling_steps"]),
                )
            )
    rows.sort(key=lambda row: int(row.manifest_index))
    return rows


def resolve_source_attempt(source_root: Path, n_train: int, seed: int, param: str) -> Path:
    seed_dir = f"seed_{seed:04d}"
    pattern = (
        f"runs/mem_gen_boundary_20260427_celeba32_N{n_train}_B*_"
        f"{param}_steps1000000_s{seed}-*/seeds/{seed_dir}/attempts/DONE_*"
    )
    matches = sorted(path for path in source_root.glob(pattern) if path.is_dir())
    if len(matches) != 1:
        rendered = "\n".join(str(path) for path in matches) or "<none>"
        raise RuntimeError(
            f"Expected exactly one completed attempt for param={param}, "
            f"N={n_train}, seed={seed}; found {len(matches)}:\n{rendered}"
        )
    return matches[0]


def targeted_manifest(source_root: Path, n_train: int, seed: int, params: list[str]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for param in params:
        attempt = resolve_source_attempt(source_root, n_train, seed, param)
        run_id = attempt.parents[3].name
        match = RUN_RE.search(run_id)
        if match is None:
            raise RuntimeError(f"Could not parse run id: {run_id}")
        config = load_config_snapshot(attempt)
        checkpoints = sorted(list_checkpoint_paths(attempt), key=checkpoint_step)
        rows.append(
            ManifestRow(
                manifest_index=len(rows),
                n_train=int(match.group("n_train")),
                seed=int(match.group("seed")),
                param=param,
                batch_size=int(match.group("batch")),
                train_steps=int(match.group("train_steps")),
                run_id=run_id,
                attempt_id=attempt.name,
                attempt_dir=str(attempt),
                checkpoint_count=len(checkpoints),
                final_step=checkpoint_step(checkpoints[-1]),
                sampler=str(getattr(config.sampling, "method", DEFAULT_SAMPLER)),
                sampling_steps=int(getattr(config.sampling, "n_steps", DEFAULT_SAMPLING_STEPS)),
            )
        )
    return rows


def select_manifest_rows(args: argparse.Namespace, manifest: list[ManifestRow]) -> list[ManifestRow]:
    indices = parse_int_list(args.manifest_indices)
    if args.manifest_index is not None:
        indices = [int(args.manifest_index)]
    elif indices is None and os.environ.get("SLURM_ARRAY_TASK_ID") and args.use_slurm_array_task_id:
        indices = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    if indices is None:
        return manifest
    by_index = {row.manifest_index: row for row in manifest}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise RuntimeError(f"Manifest index/indices not found: {missing}")
    return [by_index[index] for index in indices]


def validate_manifest(manifest: list[ManifestRow], expected_count: int | None, allow_mismatch: bool) -> None:
    if expected_count is not None and len(manifest) != expected_count and not allow_mismatch:
        raise RuntimeError(f"Manifest has {len(manifest)} rows, expected {expected_count}")
    bad_ckpts = [row for row in manifest if row.checkpoint_count != 72]
    bad_final = [row for row in manifest if row.final_step != 1_000_000]
    bad_sampler = [row for row in manifest if (row.sampler, row.sampling_steps) != ("heun", 50)]
    if (bad_ckpts or bad_final or bad_sampler) and not allow_mismatch:
        raise RuntimeError(
            "Manifest validation failed: "
            f"bad_ckpts={len(bad_ckpts)} bad_final={len(bad_final)} bad_sampler={len(bad_sampler)}"
        )


def write_manifest(out_dir: Path, manifest: list[ManifestRow]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(manifest[0]).keys()) if manifest else list(ManifestRow.__dataclass_fields__)
    csv_path = out_dir / "manifest.csv"
    jsonl_path = out_dir / "manifest.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow(asdict(row))
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def attempt_output_dir(out_dir: Path, row: ManifestRow) -> Path:
    safe_attempt = sanitize_cache_part(row.attempt_id)
    if "mem_gen_boundary_capacity_" in row.run_id:
        safe_run = sanitize_cache_part(row.run_id)
        return out_dir / "attempts" / f"m{row.manifest_index:04d}_{safe_run}_{safe_attempt}"
    return out_dir / "attempts" / f"N{row.n_train}_s{row.seed}_{row.param}_{safe_attempt}"


def select_checkpoints(
    checkpoint_paths: list[Path],
    requested_steps: list[int] | None,
    limit_checkpoints: int | None,
) -> list[Path]:
    sorted_paths = sorted(checkpoint_paths, key=checkpoint_step)
    if requested_steps:
        by_step = {checkpoint_step(path): path for path in sorted_paths}
        missing = [step for step in requested_steps if step not in by_step]
        if missing:
            raise RuntimeError(f"Requested checkpoint steps not found: {missing}")
        return [by_step[step] for step in requested_steps]
    if limit_checkpoints is None or limit_checkpoints <= 0:
        return sorted_paths
    if limit_checkpoints >= len(sorted_paths):
        return sorted_paths
    if limit_checkpoints == 1:
        return [sorted_paths[-1]]
    indices = np.linspace(0, len(sorted_paths) - 1, num=limit_checkpoints)
    unique_indices = sorted({int(round(index)) for index in indices})
    return [sorted_paths[index] for index in unique_indices]


def load_tensor(path: Path, name: str) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected {name} at {path} to be a tensor, got {type(tensor)!r}")
    return tensor


@contextmanager
def file_lock(lock_path: Path, poll_seconds: float = 5.0, stale_after_seconds: float = 14_400.0):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_after_seconds:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_completed_keys(csv_path: Path, jsonl_path: Path) -> set[EvalKey]:
    keys: set[EvalKey] = set()
    for row in read_jsonl(jsonl_path):
        key = EvalKey.from_row(row)
        if key is not None:
            keys.add(key)
    for row in read_csv_rows(csv_path):
        key = EvalKey.from_row(row)
        if key is not None:
            keys.add(key)
    return keys


def append_checkpoint_row(
    csv_path: Path,
    jsonl_path: Path,
    row: dict[str, Any],
    fieldnames: list[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def row_sort_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        safe_int(row.get("n_train")),
        safe_int(row.get("seed")),
        str(row.get("param", "")),
        safe_int(row.get("step")),
    )


def fixed_loss_columns(fixed_times: Iterable[float]) -> list[str]:
    return [f"test_loss_{time_key(float(t_value))}_10k" for t_value in fixed_times]


def rho_columns() -> list[str]:
    columns: list[str] = []
    for rho in RHO_SWEEP:
        columns.append(f"f_mem_rho{rho}")
        columns.append(f"baseline_f_mem_rho{rho}")
        columns.append(f"delta_f_mem_rho{rho}")
    return columns


def build_fieldnames(fixed_times: Iterable[float]) -> list[str]:
    return [
        "manifest_index",
        "n_train",
        "seed",
        "param",
        "prediction_space",
        "loss_space",
        "run_id",
        "attempt_id",
        "step",
        "tau",
        "f_mem",
        "baseline_f_mem",
        "delta_f_mem",
        "mean_r",
        "median_r",
        "baseline_mean_r",
        "baseline_median_r",
        *rho_columns(),
        "fid_10k",
        "frechet_inception_distance",
        "test_random_t_loss_10k",
        "test_fixed_t_loss_mean_10k",
        *fixed_loss_columns(fixed_times),
        "train_count",
        "sample_count",
        "fid_reference_count",
        "test_loss_count",
        "heldout_split",
        "heldout_source",
        "heldout_cache_path",
        "z_init_source",
        "sampler",
        "sampling_steps",
        "fid_feature_extractor",
        "fid_feature_layer",
        "fid_reference_cache_path",
        "fid_reference_cache_hit",
        "checkpoint_path",
        "source_attempt",
        "sample_runtime_sec",
        "f_mem_runtime_sec",
        "fid_runtime_sec",
        "test_loss_runtime_sec",
        "checkpoint_runtime_sec",
    ]


def make_eval_z_init(config: Any, train_tensor: torch.Tensor, sample_count: int) -> torch.Tensor:
    sample_shape = tuple(train_tensor.shape[1:])
    generator = torch.Generator().manual_seed(int(config.training.seed) + 2003)
    return torch.randn(sample_count, *sample_shape, generator=generator)


def heldout_cache_path(out_dir: Path, seed: int, split: str, count: int, config: Any) -> Path:
    image_size = int(getattr(config.data, "image_size", 32))
    channels = int(getattr(config.data, "image_channels", 1))
    return (
        out_dir
        / "cache"
        / "heldout_reference"
        / f"celeba32_grey_{split}_seed{seed}_n{count}_c{channels}_s{image_size}.pt"
    )


def load_or_build_heldout_reference(
    out_dir: Path,
    config: Any,
    seed: int,
    split: str,
    count: int,
) -> tuple[torch.Tensor, Path, str]:
    cache_path = heldout_cache_path(out_dir, seed, split, count, config)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["tensor"].contiguous(), cache_path, str(payload.get("source", f"torchvision_celeba:{split}"))

    with file_lock(lock_path):
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            return payload["tensor"].contiguous(), cache_path, str(payload.get("source", f"torchvision_celeba:{split}"))
        dataset = datasets.CelebA(
            root=config.data.root,
            split=split,
            target_type="identity",
            transform=_build_transform(),
            download=bool(config.data.download),
        )
        indices = seeded_indices(len(dataset), count, int(config.training.seed) + 1)
        tensor, labels = _stack_subset(dataset, indices)
        payload = {
            "tensor": tensor.contiguous(),
            "labels": labels,
            "indices": indices,
            "source": f"torchvision_celeba:{split}",
            "seed": int(seed),
            "selection_seed": int(config.training.seed) + 1,
            "count": int(count),
            "preprocessing": "center_crop_140_resize_32_grayscale_normalize_minus1_1",
        }
        atomic_torch_save(payload, cache_path)
        return tensor.contiguous(), cache_path, f"torchvision_celeba:{split}"


def verify_deterministic_inputs(
    row: ManifestRow,
    attempt_dir: Path,
    z_eval: torch.Tensor,
    heldout_tensor: torch.Tensor,
) -> None:
    artifact_dir = attempt_dir / "artifacts"
    saved_z_path = artifact_dir / "z_init.pt"
    saved_reference_path = artifact_dir / "reference_tensor.pt"
    if saved_z_path.exists():
        saved_z = torch.load(saved_z_path, map_location="cpu")
        compare_count = min(int(saved_z.shape[0]), int(z_eval.shape[0]))
        if not torch.equal(z_eval[:compare_count], saved_z[:compare_count]):
            raise RuntimeError(f"Generated z_init prefix does not match saved z_init for {row.run_id}")
    if saved_reference_path.exists():
        saved_reference = torch.load(saved_reference_path, map_location="cpu")
        compare_count = min(int(saved_reference.shape[0]), int(heldout_tensor.shape[0]))
        if not torch.equal(heldout_tensor[:compare_count], saved_reference[:compare_count]):
            raise RuntimeError(f"Rebuilt heldout prefix does not match saved reference for {row.run_id}")


def sample_in_batches(
    model: GDFM,
    z_init: torch.Tensor,
    config: Any,
    device: torch.device,
    batch_size: int,
    sampler: str,
    sampling_steps: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, z_init.shape[0], batch_size):
            batch = z_init[start : start + batch_size].to(device, non_blocking=True)
            with autocast_context(config, device):
                samples = model.sample(batch, n_steps=sampling_steps, method=sampler, device=device)
            outputs.append(samples.detach().cpu().float())
    if was_training:
        model.train()
    return torch.cat(outputs, dim=0)


def covariance_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={values.shape}")
    mu = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        sigma = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(values, rowvar=False)
    return mu, sigma


def fid_from_stats(
    sample_mu: np.ndarray,
    sample_sigma: np.ndarray,
    ref_mu: np.ndarray,
    ref_sigma: np.ndarray,
    eps: float = 1e-6,
) -> float:
    sample_mu = np.asarray(sample_mu, dtype=np.float64)
    ref_mu = np.asarray(ref_mu, dtype=np.float64)
    sample_sigma = np.asarray(sample_sigma, dtype=np.float64)
    ref_sigma = np.asarray(ref_sigma, dtype=np.float64)
    diff = sample_mu - ref_mu
    covmean, _ = linalg.sqrtm(sample_sigma @ ref_sigma, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sample_sigma.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sample_sigma + offset) @ (ref_sigma + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sample_sigma + ref_sigma - 2.0 * covmean)
    return float(max(fid, 0.0))


def sanitize_cache_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def fid_layer_name(encoder: _TorchFidelityFeatureEncoder) -> str:
    return str(getattr(encoder, "layer_name", getattr(encoder, "feature_layer", "default")))


def fid_reference_cache_path(
    out_dir: Path,
    seed: int,
    split: str,
    reference_count: int,
    feature_extractor: str,
    layer: str,
) -> Path:
    return (
        out_dir
        / "cache"
        / "fid_reference"
        / (
            f"celeba32_grey_{split}_seed{seed}_n{reference_count}_"
            f"{sanitize_cache_part(feature_extractor)}_{sanitize_cache_part(layer)}.pt"
        )
    )


def load_or_compute_fid_reference(
    out_dir: Path,
    seed: int,
    split: str,
    reference_tensor: torch.Tensor,
    reference_count: int,
    encoder: _TorchFidelityFeatureEncoder,
    feature_extractor: str,
    batch_size: int,
) -> FidReference:
    layer = fid_layer_name(encoder)
    cache_path = fid_reference_cache_path(out_dir, seed, split, reference_count, feature_extractor, layer)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return FidReference(
            mu=np.asarray(payload["mu"], dtype=np.float64),
            sigma=np.asarray(payload["sigma"], dtype=np.float64),
            count=int(payload["count"]),
            feature_extractor=str(payload["feature_extractor"]),
            feature_layer=str(payload["feature_layer"]),
            cache_path=cache_path,
            from_cache=True,
        )

    with file_lock(lock_path):
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            return FidReference(
                mu=np.asarray(payload["mu"], dtype=np.float64),
                sigma=np.asarray(payload["sigma"], dtype=np.float64),
                count=int(payload["count"]),
                feature_extractor=str(payload["feature_extractor"]),
                feature_layer=str(payload["feature_layer"]),
                cache_path=cache_path,
                from_cache=True,
            )
        reference = reference_tensor[:reference_count].contiguous()
        features = _encode_with_torch_fidelity_encoder(
            reference,
            encoder=encoder,
            batch_size=batch_size,
            image_range=FID_IMAGE_RANGE,
        )
        mu, sigma = covariance_stats(features)
        payload = {
            "mu": mu,
            "sigma": sigma,
            "count": int(reference.shape[0]),
            "feature_extractor": feature_extractor,
            "feature_layer": layer,
            "seed": int(seed),
            "split": split,
        }
        atomic_torch_save(payload, cache_path)
        return FidReference(
            mu=mu,
            sigma=sigma,
            count=int(reference.shape[0]),
            feature_extractor=feature_extractor,
            feature_layer=layer,
            cache_path=cache_path,
            from_cache=False,
        )


def compute_fid_with_reference(
    samples: torch.Tensor,
    reference: FidReference,
    encoder: _TorchFidelityFeatureEncoder,
    batch_size: int,
) -> float:
    features = _encode_with_torch_fidelity_encoder(
        samples,
        encoder=encoder,
        batch_size=batch_size,
        image_range=FID_IMAGE_RANGE,
    )
    sample_mu, sample_sigma = covariance_stats(features)
    return fid_from_stats(sample_mu, sample_sigma, reference.mu, reference.sigma)


def rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return [index]


def weighted_mean(values: list[tuple[float, int]]) -> float:
    total_count = sum(count for _, count in values)
    if total_count <= 0:
        return math.nan
    return float(sum(value * count for value, count in values) / total_count)


def compute_test_losses(
    model: GDFM,
    reference_tensor: torch.Tensor,
    config: Any,
    device: torch.device,
    step: int,
    batch_size: int,
    fixed_times: list[float],
) -> dict[str, float | int]:
    seed = int(getattr(config, "seed", getattr(config.training, "seed", 0)) or 0)
    was_training = model.training
    model.eval()
    random_losses: list[tuple[float, int]] = []
    fixed_losses: dict[float, list[tuple[float, int]]] = {float(t): [] for t in fixed_times}
    devices = rng_devices(device)

    with torch.no_grad(), torch.random.fork_rng(devices=devices, enabled=True):
        for batch_index, start in enumerate(range(0, reference_tensor.shape[0], batch_size)):
            batch_cpu = reference_tensor[start : start + batch_size]
            batch = batch_cpu.to(device, non_blocking=True)
            count = int(batch.shape[0])

            torch.manual_seed(seed + int(step) + 1009 + batch_index)
            with autocast_context(config, device):
                loss = model.compute_loss(batch)
            random_losses.append((float(loss.detach().cpu()), count))

            for fixed_t in fixed_times:
                t_value = float(fixed_t)
                torch.manual_seed(seed + int(step) + int(round(t_value * 10000.0)) + 2027 + batch_index)
                t = torch.full((count,), t_value, device=device, dtype=torch.float32)
                eps = torch.randn_like(batch)
                with autocast_context(config, device):
                    fixed_loss = model.compute_loss(batch, eps=eps, t=t)
                fixed_losses[t_value].append((float(fixed_loss.detach().cpu()), count))

    if was_training:
        model.train()

    result: dict[str, float | int] = {
        "test_random_t_loss_10k": weighted_mean(random_losses),
        "test_loss_count": int(reference_tensor.shape[0]),
    }
    fixed_means: list[float] = []
    for fixed_t in fixed_times:
        mean = weighted_mean(fixed_losses[float(fixed_t)])
        fixed_means.append(mean)
        result[f"test_loss_{time_key(float(fixed_t))}_10k"] = mean
    result["test_fixed_t_loss_mean_10k"] = float(np.nanmean(fixed_means)) if fixed_means else math.nan
    return result


def build_metric_row(
    manifest_row: ManifestRow,
    step: int,
    source_attempt: Path,
    checkpoint_path: Path,
    fmem_record: dict[str, Any],
    fid_value: float,
    loss_record: dict[str, Any],
    fid_reference: FidReference,
    train_count: int,
    sample_count: int,
    fid_reference_count: int,
    heldout_split: str,
    heldout_source: str,
    heldout_cache_path_value: Path,
    z_init_source: str,
    sampler: str,
    sampling_steps: int,
    runtime_record: dict[str, float],
) -> dict[str, Any]:
    prediction_space, loss_space = PARAM_SPACES[manifest_row.param]
    row: dict[str, Any] = {
        "manifest_index": int(manifest_row.manifest_index),
        "n_train": int(manifest_row.n_train),
        "seed": int(manifest_row.seed),
        "param": manifest_row.param,
        "prediction_space": prediction_space,
        "loss_space": loss_space,
        "run_id": manifest_row.run_id,
        "attempt_id": manifest_row.attempt_id,
        "step": int(step),
        "tau": float(step) / 1_000_000.0,
        "f_mem": fmem_record.get("f_mem"),
        "baseline_f_mem": fmem_record.get("baseline_f_mem"),
        "delta_f_mem": fmem_record.get("delta_f_mem"),
        "mean_r": fmem_record.get("mean_r"),
        "median_r": fmem_record.get("median_r"),
        "baseline_mean_r": fmem_record.get("baseline_mean_r"),
        "baseline_median_r": fmem_record.get("baseline_median_r"),
        "fid_10k": fid_value,
        "frechet_inception_distance": fid_value,
        "train_count": int(train_count),
        "sample_count": int(sample_count),
        "fid_reference_count": int(fid_reference_count),
        "heldout_split": heldout_split,
        "heldout_source": heldout_source,
        "heldout_cache_path": str(heldout_cache_path_value),
        "z_init_source": z_init_source,
        "sampler": sampler,
        "sampling_steps": int(sampling_steps),
        "fid_feature_extractor": fid_reference.feature_extractor,
        "fid_feature_layer": fid_reference.feature_layer,
        "fid_reference_cache_path": str(fid_reference.cache_path),
        "fid_reference_cache_hit": bool(fid_reference.from_cache),
        "checkpoint_path": str(checkpoint_path),
        "source_attempt": str(source_attempt),
    }
    for rho in RHO_SWEEP:
        row[f"f_mem_rho{rho}"] = fmem_record.get(f"f_mem_rho{rho}")
        row[f"baseline_f_mem_rho{rho}"] = fmem_record.get(f"baseline_f_mem_rho{rho}")
        row[f"delta_f_mem_rho{rho}"] = fmem_record.get(f"delta_f_mem_rho{rho}")
    row.update(loss_record)
    row.update(runtime_record)
    return row


def read_metric_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(csv_path)
    rows.sort(key=row_sort_key)
    return rows


def plot_dual_axis(
    rows: list[dict[str, Any]],
    title: str,
    right_column: str,
    right_label: str,
    out_path: Path,
) -> None:
    if not rows:
        return
    tau = [safe_float(row.get("tau")) for row in rows]
    f_mem = [safe_float(row.get("f_mem")) for row in rows]
    right = [safe_float(row.get(right_column)) for row in rows]
    fig, ax_left = plt.subplots(figsize=(8.0, 4.8))
    ax_right = ax_left.twinx()
    ax_left.plot(tau, f_mem, color="#204a87", marker="o", markersize=3, linewidth=1.5)
    ax_right.plot(tau, right, color="#a40000", marker="s", markersize=3, linewidth=1.3)
    ax_left.set_xlabel("tau = step / 1,000,000")
    ax_left.set_ylabel("f_mem", color="#204a87")
    ax_right.set_ylabel(right_label, color="#a40000")
    ax_left.tick_params(axis="y", labelcolor="#204a87")
    ax_right.tick_params(axis="y", labelcolor="#a40000")
    ax_left.grid(True, alpha=0.25)
    ax_left.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_attempt_outputs(attempt_dir: Path, row: ManifestRow) -> None:
    rows = read_metric_rows(attempt_dir / "checkpoint_metrics.csv")
    label = f"N={row.n_train} seed={row.seed} {row.param}"
    plot_dual_axis(
        rows,
        title=f"{label}: f_mem vs 10K held-out test loss",
        right_column="test_random_t_loss_10k",
        right_label="10K held-out random-t loss",
        out_path=attempt_dir / "f_mem_vs_test_loss_tau.png",
    )
    plot_dual_axis(
        rows,
        title=f"{label}: f_mem vs FID@10K",
        right_column="fid_10k",
        right_label="FID@10K",
        out_path=attempt_dir / "f_mem_vs_fid_tau.png",
    )


def plot_grouped_metric(
    rows: list[dict[str, Any]],
    group_rows: list[ManifestRow],
    metric_column: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(8.2, 4.9))
    for manifest_row in group_rows:
        series = [
            row
            for row in rows
            if safe_int(row.get("n_train")) == manifest_row.n_train
            and safe_int(row.get("seed")) == manifest_row.seed
            and row.get("param") == manifest_row.param
        ]
        series.sort(key=lambda item: safe_int(item.get("step")))
        if not series:
            continue
        tau = [safe_float(item.get("tau")) for item in series]
        values = [safe_float(item.get(metric_column)) for item in series]
        axis.plot(tau, values, marker="o", markersize=2.5, linewidth=1.2, label=manifest_row.param)
    axis.set_xlabel("tau = step / 1,000,000")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_by_n(
    final_rows: list[dict[str, Any]],
    metric_column: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if not final_rows:
        return
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    n_values = sorted({safe_int(row.get("n_train")) for row in final_rows})
    for param in PARAM_SPACES:
        means: list[float] = []
        for n_train in n_values:
            vals = [
                safe_float(row.get(metric_column))
                for row in final_rows
                if row.get("param") == param and safe_int(row.get("n_train")) == n_train
            ]
            vals = [value for value in vals if math.isfinite(value)]
            means.append(float(np.mean(vals)) if vals else math.nan)
        axis.plot(n_values, means, marker="o", linewidth=1.5, label=param)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("n_train")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def refresh_final_csv(attempt_dir: Path, fieldnames: list[str]) -> list[dict[str, Any]]:
    rows = read_metric_rows(attempt_dir / "checkpoint_metrics.csv")
    if not rows:
        write_csv(attempt_dir / "final_metrics.csv", [], fieldnames)
        return []
    max_step = max(safe_int(row.get("step")) for row in rows)
    final_rows = [row for row in rows if safe_int(row.get("step")) == max_step]
    write_csv(attempt_dir / "final_metrics.csv", final_rows[:1], fieldnames)
    return final_rows[:1]


def rebuild_combined_outputs(out_dir: Path, manifest: list[ManifestRow], fieldnames: list[str]) -> None:
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for row in manifest:
        attempt_dir = attempt_output_dir(out_dir, row)
        rows = read_metric_rows(attempt_dir / "checkpoint_metrics.csv")
        if rows:
            all_rows.extend(rows)
        finals = refresh_final_csv(attempt_dir, fieldnames)
        final_rows.extend(finals)
        plot_attempt_outputs(attempt_dir, row)

    all_rows.sort(key=row_sort_key)
    final_rows.sort(key=row_sort_key)
    write_csv(out_dir / "all_checkpoint_metrics.csv", all_rows, fieldnames)
    write_csv(out_dir / "all_final_metrics.csv", final_rows, fieldnames)

    by_n_seed: dict[tuple[int, int], list[ManifestRow]] = {}
    for row in manifest:
        by_n_seed.setdefault((row.n_train, row.seed), []).append(row)
    for (n_train, seed), group_rows in sorted(by_n_seed.items()):
        group_rows = sorted(group_rows, key=lambda item: PARAM_ORDER[item.param])
        prefix = out_dir / "plots" / "by_n_seed" / f"N{n_train}_seed{seed}"
        plot_grouped_metric(
            all_rows,
            group_rows,
            metric_column="f_mem",
            ylabel="f_mem",
            title=f"N={n_train} seed={seed}: f_mem over training time",
            out_path=prefix.with_name(prefix.name + "_f_mem_tau.png"),
        )
        plot_grouped_metric(
            all_rows,
            group_rows,
            metric_column="fid_10k",
            ylabel="FID@10K",
            title=f"N={n_train} seed={seed}: FID@10K over training time",
            out_path=prefix.with_name(prefix.name + "_fid_tau.png"),
        )
        plot_grouped_metric(
            all_rows,
            group_rows,
            metric_column="test_random_t_loss_10k",
            ylabel="10K held-out random-t loss",
            title=f"N={n_train} seed={seed}: test loss over training time",
            out_path=prefix.with_name(prefix.name + "_test_loss_tau.png"),
        )

    plot_final_by_n(
        final_rows,
        metric_column="f_mem",
        ylabel="final f_mem",
        title="Final f_mem by train size, averaged over seeds",
        out_path=out_dir / "compare_final_f_mem_by_n_train.png",
    )
    plot_final_by_n(
        final_rows,
        metric_column="fid_10k",
        ylabel="final FID@10K",
        title="Final FID@10K by train size, averaged over seeds",
        out_path=out_dir / "compare_final_fid_by_n_train.png",
    )
    plot_final_by_n(
        final_rows,
        metric_column="test_random_t_loss_10k",
        ylabel="final 10K held-out random-t loss",
        title="Final test loss by train size, averaged over seeds",
        out_path=out_dir / "compare_final_test_loss_by_n_train.png",
    )


def evaluate_manifest_row(args: argparse.Namespace, manifest_row: ManifestRow, fieldnames: list[str]) -> None:
    out_dir = Path(args.out_dir)
    source_attempt = Path(manifest_row.attempt_dir)
    eval_attempt_dir = attempt_output_dir(out_dir, manifest_row)
    eval_attempt_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_snapshot(source_attempt)
    device = torch.device(args.device)
    configure_torch_runtime(config, device)

    checkpoint_paths = sorted(list_checkpoint_paths(source_attempt), key=checkpoint_step)
    selected_checkpoints = select_checkpoints(
        checkpoint_paths,
        requested_steps=parse_int_list(args.checkpoint_steps),
        limit_checkpoints=args.limit_checkpoints,
    )
    if len(checkpoint_paths) != 72 and not args.allow_checkpoint_count_mismatch:
        raise RuntimeError(
            f"{manifest_row.run_id} resolved {len(checkpoint_paths)} checkpoints, expected 72. "
            "Pass --allow_checkpoint_count_mismatch for prototype inputs."
        )
    if checkpoint_step(checkpoint_paths[-1]) != 1_000_000 and not args.allow_checkpoint_count_mismatch:
        raise RuntimeError(f"{manifest_row.run_id} final checkpoint is not step_1000000.pt")

    artifact_dir = source_attempt / "artifacts"
    train_tensor = load_tensor(artifact_dir / "train_tensor.pt", "train tensor")

    sample_count = int(args.sample_count)
    fid_reference_count = int(args.fid_reference_count)
    heldout_count = max(sample_count, fid_reference_count)
    heldout_tensor, heldout_path, heldout_source = load_or_build_heldout_reference(
        out_dir=out_dir,
        config=config,
        seed=manifest_row.seed,
        split=str(args.heldout_split),
        count=heldout_count,
    )
    if tuple(heldout_tensor.shape[1:]) != tuple(train_tensor.shape[1:]):
        raise RuntimeError(
            f"Heldout shape {tuple(heldout_tensor.shape)} incompatible with train shape {tuple(train_tensor.shape)}"
        )
    z_eval = make_eval_z_init(config, train_tensor, sample_count)
    if args.verify_deterministic_inputs:
        verify_deterministic_inputs(manifest_row, source_attempt, z_eval, heldout_tensor)

    sampler = str(args.sampler or manifest_row.sampler)
    sampling_steps = int(args.sampling_steps or manifest_row.sampling_steps)
    completed_keys = set() if args.overwrite else load_completed_keys(
        eval_attempt_dir / "checkpoint_metrics.csv",
        eval_attempt_dir / "checkpoint_metrics.jsonl",
    )

    remaining: list[Path] = []
    for checkpoint_path in selected_checkpoints:
        key = EvalKey(
            n_train=manifest_row.n_train,
            seed=manifest_row.seed,
            param=manifest_row.param,
            step=checkpoint_step(checkpoint_path),
            sample_count=sample_count,
            fid_reference_count=fid_reference_count,
            heldout_split=str(args.heldout_split),
            sampler=sampler,
            sampling_steps=sampling_steps,
        )
        if key not in completed_keys:
            remaining.append(checkpoint_path)

    print(
        f"[manifest {manifest_row.manifest_index}] N={manifest_row.n_train} seed={manifest_row.seed} "
        f"param={manifest_row.param} selected={len(selected_checkpoints)} remaining={len(remaining)} "
        f"out={eval_attempt_dir}",
        flush=True,
    )
    if not remaining:
        refresh_final_csv(eval_attempt_dir, fieldnames)
        plot_attempt_outputs(eval_attempt_dir, manifest_row)
        return

    reference_for_fmem = heldout_tensor[:sample_count].contiguous()
    reference_for_loss = heldout_tensor[:sample_count].contiguous()
    feature_extractor = str(args.fid_feature_extractor)
    encoder = _TorchFidelityFeatureEncoder(f"torch_fidelity:{feature_extractor}", device=device)
    fid_reference = load_or_compute_fid_reference(
        out_dir=out_dir,
        seed=manifest_row.seed,
        split=str(args.heldout_split),
        reference_tensor=heldout_tensor,
        reference_count=fid_reference_count,
        encoder=encoder,
        feature_extractor=feature_extractor,
        batch_size=int(args.fid_batch_size),
    )

    model = GDFM(config).to(device)
    fixed_times = [float(t) for t in loss_eval_times(config)]
    train_count = int(train_tensor.shape[0])

    for checkpoint_path in remaining:
        step = checkpoint_step(checkpoint_path)
        checkpoint_start = time.time()
        print(f"[{manifest_row.manifest_index}] evaluating step={step} checkpoint={checkpoint_path.name}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"])

        sample_start = time.time()
        samples = sample_in_batches(
            model,
            z_eval,
            config=config,
            device=device,
            batch_size=int(args.sample_batch_size),
            sampler=sampler,
            sampling_steps=sampling_steps,
        )
        sample_runtime = time.time() - sample_start

        fmem_start = time.time()
        fmem_record = collect_image32_metric_record(
            step=step,
            samples=samples,
            baseline_samples=reference_for_fmem,
            train_tensor=train_tensor,
            chunk_size=int(getattr(config.evaluation, "chunk_size", 2048)),
            config=config,
        )
        fmem_runtime = time.time() - fmem_start

        fid_start = time.time()
        fid_value = compute_fid_with_reference(
            samples=samples,
            reference=fid_reference,
            encoder=encoder,
            batch_size=int(args.fid_batch_size),
        )
        fid_runtime = time.time() - fid_start

        loss_start = time.time()
        loss_record = compute_test_losses(
            model=model,
            reference_tensor=reference_for_loss,
            config=config,
            device=device,
            step=step,
            batch_size=int(args.loss_batch_size),
            fixed_times=fixed_times,
        )
        loss_runtime = time.time() - loss_start

        runtime_record = {
            "sample_runtime_sec": sample_runtime,
            "f_mem_runtime_sec": fmem_runtime,
            "fid_runtime_sec": fid_runtime,
            "test_loss_runtime_sec": loss_runtime,
            "checkpoint_runtime_sec": time.time() - checkpoint_start,
        }
        row = build_metric_row(
            manifest_row=manifest_row,
            step=step,
            source_attempt=source_attempt,
            checkpoint_path=checkpoint_path,
            fmem_record=fmem_record,
            fid_value=fid_value,
            loss_record=loss_record,
            fid_reference=fid_reference,
            train_count=train_count,
            sample_count=sample_count,
            fid_reference_count=fid_reference_count,
            heldout_split=str(args.heldout_split),
            heldout_source=heldout_source,
            heldout_cache_path_value=heldout_path,
            z_init_source=f"deterministic_seed_{int(config.training.seed) + 2003}_count_{sample_count}",
            sampler=sampler,
            sampling_steps=sampling_steps,
            runtime_record=runtime_record,
        )
        append_checkpoint_row(
            eval_attempt_dir / "checkpoint_metrics.csv",
            eval_attempt_dir / "checkpoint_metrics.jsonl",
            row,
            fieldnames,
        )
        print(
            f"[{manifest_row.manifest_index}] wrote step={step} "
            f"f_mem={safe_float(row.get('f_mem')):.6g} "
            f"fid_10k={safe_float(row.get('fid_10k')):.6g} "
            f"test_loss={safe_float(row.get('test_random_t_loss_10k')):.6g}",
            flush=True,
        )

    refresh_final_csv(eval_attempt_dir, fieldnames)
    plot_attempt_outputs(eval_attempt_dir, manifest_row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest_csv", type=Path, default=None)
    parser.add_argument("--all_done", action="store_true")
    parser.add_argument("--expected_manifest_count", type=int, default=84)
    parser.add_argument("--n_train", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", type=str, default=DEFAULT_PARAMS)
    parser.add_argument("--sample_count", type=int, default=10_000)
    parser.add_argument("--fid_reference_count", type=int, default=10_000)
    parser.add_argument("--heldout_split", type=str, default=DEFAULT_HELDOUT_SPLIT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default=None, choices=("euler", "heun"))
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--fid_batch_size", type=int, default=128)
    parser.add_argument("--loss_batch_size", type=int, default=256)
    parser.add_argument("--fid_feature_extractor", type=str, default=DEFAULT_FID_FEATURE_EXTRACTOR)
    parser.add_argument("--checkpoint_steps", type=str, default=None)
    parser.add_argument("--limit_checkpoints", type=int, default=None)
    parser.add_argument("--manifest_index", type=int, default=None)
    parser.add_argument("--manifest_indices", type=str, default=None)
    parser.add_argument("--use_slurm_array_task_id", action="store_true")
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--write_manifest", action="store_true")
    parser.add_argument("--combine_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_checkpoint_count_mismatch", action="store_true")
    parser.add_argument("--verify_deterministic_inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_combined", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_root = Path(args.source_root)
    if args.manifest_csv is not None:
        manifest = load_manifest_csv(Path(args.manifest_csv))
        validate_manifest(manifest, args.expected_manifest_count, args.allow_checkpoint_count_mismatch)
    elif args.all_done:
        manifest = discover_done_manifest(source_root)
        validate_manifest(manifest, args.expected_manifest_count, args.allow_checkpoint_count_mismatch)
    else:
        manifest = targeted_manifest(source_root, args.n_train, args.seed, parse_params(args.params))
        validate_manifest(manifest, None, args.allow_checkpoint_count_mismatch)

    out_dir = Path(args.out_dir)
    if args.write_manifest or args.manifest_only or args.combine_only:
        write_manifest(out_dir, manifest)
    if args.manifest_only:
        print(f"wrote manifest rows={len(manifest)} out={out_dir}", flush=True)
        return 0

    field_config = load_config_snapshot(Path(manifest[0].attempt_dir))
    fieldnames = build_fieldnames(loss_eval_times(field_config))
    selected_rows = [] if args.combine_only else select_manifest_rows(args, manifest)

    for row in selected_rows:
        evaluate_manifest_row(args, row, fieldnames)

    if not args.skip_combined:
        rebuild_combined_outputs(out_dir, manifest, fieldnames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
