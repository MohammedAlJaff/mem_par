"""
Sanity Check 4: JiT memory profiling per resolution.

Before launching expensive 128/256 sweeps, verify that JiT-B/16 fits
in GPU memory at each resolution. This script answers a practical question:
"what is the largest batch size that survives a forward/backward pass here?"

This is especially important for 256×256 where seq_len=256 patches
makes self-attention expensive.

Pedagogical note:
    A memory sanity check is not a quality check. Passing here only means
    the model fits and produces tensors of the right shape; it says nothing
    about whether the model trains well or samples well.

Usage:
    python lab/sanity/check_jit_memory.py --device cuda

NOTE: This script requires the JiT submodule to be set up.
If JiT is not yet integrated, it will print instructions and exit.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_memory(device: str = "cuda"):
    """Profile the largest trainable batch size for JiT at several resolutions."""
    print("=" * 60)
    print("SANITY CHECK: JiT memory profiling")
    print("=" * 60)

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        print(f"  Device '{device}' is not CUDA. Skipping memory check.")
        print("  This check is only meaningful on GPU.")
        return

    if not torch.cuda.is_available():
        print("  CUDA not available. Skipping memory check.")
        print("  This check is only meaningful on GPU.")
        return

    device_index = (
        torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    )
    gpu_name = torch.cuda.get_device_name(device_index)
    gpu_mem = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # If JiT is unavailable, we still do a lighter-weight tensor-plumbing
    # check so the script remains useful during early integration work.
    try:
        from experiments.natural_images.common.networks import JiTWrapper
        print("  JiT wrapper loaded successfully.")
    except (ImportError, NotImplementedError, FileNotFoundError) as e:
        print(f"\n  JiT not available: {e}")
        print("  To set up JiT:")
        print("    place the JiT baseline under baselines/jit/vendor")
        print("    # Then edit experiments/natural_images/common/networks/jit_wrapper.py to match their API")
        print("\n  Skipping JiT memory check. Running shape-only test instead.")
        _run_shape_only_test(device)
        return

    resolutions = [64, 128, 256]
    patch_size = 16

    for res in resolutions:
        print(f"\n  --- {res}×{res}, patch_size={patch_size} ---")
        seq_len = (res // patch_size) ** 2
        print(f"  Patches: {seq_len}, D_patch: {patch_size**2 * 3}")

        try:
            model = JiTWrapper(
                image_size=res, patch_size=patch_size,
                hidden_dim=768, depth=12, num_heads=12,
            ).to(device)
        except NotImplementedError:
            print("  JiT wrapper not fully implemented yet. Skipping.")
            continue

        # Batch size is monotone with memory: if batch size B fails, any larger
        # one should also fail. That makes binary search a cheap probe.
        max_bs, peak_gb = _find_max_batch_size(model, res, device, device_index)
        print(f"  Max batch size (fp32): {max_bs}")
        if peak_gb is not None:
            print(f"  Peak allocated memory at that batch size: {peak_gb:.2f} GB")

        del model
        torch.cuda.empty_cache()

    print(f"\n  ✓ Memory profiling complete")


def _find_max_batch_size(model, resolution, device, device_index, max_try=512):
    """
    Binary search for the largest batch size that survives forward+backward.

    Returns:
        `(best_batch_size, peak_allocated_gb_for_best_batch)`
    """
    low, high = 1, max_try
    best = 0
    best_peak_gb = None

    while low <= high:
        mid = (low + high) // 2
        x = t = out = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_index)
            x = torch.randn(mid, 3, resolution, resolution, device=device)
            t = torch.rand(mid, device=device)
            out = model(x, t)
            loss = out.mean()
            loss.backward()
            best = mid
            best_peak_gb = torch.cuda.max_memory_allocated(device_index) / (1024**3)
            low = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # CUDA OOM sometimes surfaces as a RuntimeError depending on which
            # kernel failed, so we treat both forms as "too large to fit".
            if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                raise
            high = mid - 1
        finally:
            del x, t, out, loss
            torch.cuda.empty_cache()

    return best, best_peak_gb


def _run_shape_only_test(device):
    """
    Fallback: use a UNet forward pass as a proxy tensor-shape check.

    This does not estimate JiT memory usage; it only verifies that the image
    stack is wired correctly enough to instantiate and run.
    """
    from experiments.natural_images.common.networks import make_unet_for_resolution

    print("\n  Running shape-only test with UNet as proxy:")
    for res, ch in [(64, 3), (128, 3), (256, 3)]:
        unet = make_unet_for_resolution(res, in_channels=ch).to(device)
        x = torch.randn(2, ch, res, res, device=device)
        t = torch.rand(2, device=device)
        with torch.no_grad():
            out = unet(x, t)
        ok = out.shape == x.shape
        print(f"  {res}×{res}: output shape {out.shape} {'✓' if ok else '✗'}")
        del unet, x, t, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe JiT GPU memory use before launching expensive image sweeps."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on. This check is only meaningful on CUDA.",
    )
    args = parser.parse_args()
    check_memory(args.device)
