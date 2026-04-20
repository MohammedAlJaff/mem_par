"""Small runtime knobs for numerical mode and host-to-device movement."""

from contextlib import nullcontext

import torch


PRECISION_CHOICES = ("fp32", "tf32", "bf16")


def torch_device(device) -> torch.device:
    """Normalize the runner's device argument into a torch.device."""
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def validate_precision(config, device) -> str:
    """Fail early when a requested precision is not valid on this device."""
    precision = str(getattr(getattr(config, "runtime", None), "precision", "fp32"))
    if precision not in PRECISION_CHOICES:
        raise ValueError(
            f"runtime.precision must be one of {PRECISION_CHOICES}, got {precision!r}"
        )

    torch_dev = torch_device(device)
    if precision == "bf16":
        if torch_dev.type != "cuda":
            raise ValueError("precision='bf16' requires a CUDA device.")
        if not torch.cuda.is_available():
            raise ValueError("precision='bf16' requires torch.cuda.is_available().")
        if not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            raise ValueError("precision='bf16' requires CUDA BF16 support.")
    return precision


def configure_torch_runtime(config, device):
    """
    Apply global torch runtime switches and record the resolved metadata.

    TF32 is a CUDA backend flag, while BF16 is expressed through local autocast
    contexts around forward/sample/loss code paths.
    """
    precision = validate_precision(config, device)
    torch_dev = torch_device(device)
    is_cuda = torch_dev.type == "cuda"

    tf32_enabled = bool(is_cuda and precision in {"tf32", "bf16"})
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")

    runtime = getattr(config, "runtime", None)
    pin_requested = bool(getattr(runtime, "pin_memory", True))
    pin_memory = bool(is_cuda and pin_requested)
    non_blocking_h2d = bool(pin_memory)
    autocast_dtype = "bfloat16" if precision == "bf16" else "none"

    metadata = {
        "precision": precision,
        "tf32_enabled": tf32_enabled,
        "autocast_dtype": autocast_dtype,
        "pin_memory": pin_memory,
        "non_blocking_h2d": non_blocking_h2d,
    }
    if runtime is not None:
        runtime.precision = precision
        runtime.tf32_enabled = tf32_enabled
        runtime.autocast_dtype = autocast_dtype
        runtime.pin_memory = pin_memory
        runtime.non_blocking_h2d = non_blocking_h2d
    return metadata


def autocast_context(config, device):
    """Return the BF16 autocast context requested by config, or a no-op."""
    precision = str(getattr(getattr(config, "runtime", None), "precision", "fp32"))
    torch_dev = torch_device(device)
    if precision == "bf16" and torch_dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


__all__ = [
    "PRECISION_CHOICES",
    "autocast_context",
    "configure_torch_runtime",
    "torch_device",
    "validate_precision",
]
