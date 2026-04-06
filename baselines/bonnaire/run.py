"""Thin CLI wrapper around the vendored Bonnaire baseline scripts."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
SCRIPT_PATHS = {
    "train-gmm": VENDOR_ROOT / "Experiments" / "src" / "Training" / "run_GMM.py",
    "train-unet": VENDOR_ROOT / "Experiments" / "src" / "Training" / "run_Unet.py",
    "compute-fid": VENDOR_ROOT / "Experiments" / "src" / "Evaluation" / "compute_FID.py",
    "compute-fmem": VENDOR_ROOT / "Experiments" / "src" / "Evaluation" / "compute_fmem.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a vendored Bonnaire baseline script.")
    parser.add_argument("command", choices=sorted(SCRIPT_PATHS))
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not VENDOR_ROOT.exists():
        raise FileNotFoundError(f"Vendored Bonnaire baseline missing at {VENDOR_ROOT}")

    script_path = SCRIPT_PATHS[args.command]
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    forwarded_args = list(args.script_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]
    sys.argv = [str(script_path), *forwarded_args]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
