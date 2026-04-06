"""Thin CLI wrapper around the vendored JiT baseline."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
SCRIPT_PATHS = {
    "main": VENDOR_ROOT / "main_jit.py",
    "prepare-ref": VENDOR_ROOT / "prepare_ref.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a vendored JiT baseline script.")
    parser.add_argument("command", choices=sorted(SCRIPT_PATHS))
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not VENDOR_ROOT.exists():
        raise FileNotFoundError(f"Vendored JiT baseline missing at {VENDOR_ROOT}")

    script_path = SCRIPT_PATHS[args.command]
    vendor_package_root = str(VENDOR_ROOT)
    if vendor_package_root not in sys.path:
        sys.path.insert(0, vendor_package_root)
    forwarded_args = list(args.script_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]
    sys.argv = [str(script_path), *forwarded_args]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
