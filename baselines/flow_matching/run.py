"""Launch vendored flow-matching scripts from the reorganized baseline layout."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
SCRIPT_PATHS = {
    "train-2d": VENDOR_ROOT / "scripts" / "train_flow_matching_2d.py",
    "train-image": VENDOR_ROOT / "scripts" / "train_flow_matching_on_image.py",
    "reflow-2d": VENDOR_ROOT / "scripts" / "train_reflow_2d.py",
    "compare-2d": VENDOR_ROOT / "scripts" / "plot_comparison_2d.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vendored flow-matching scripts from the baseline wrapper."
    )
    parser.add_argument("command", choices=sorted(SCRIPT_PATHS))
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def has_output_dir_arg(args: list[str]) -> bool:
    return any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in args)


def main() -> None:
    args = parse_args()
    if not VENDOR_ROOT.exists():
        raise FileNotFoundError(f"Vendored flow-matching baseline missing at {VENDOR_ROOT}")

    script_path = SCRIPT_PATHS[args.command]
    forwarded_args = list(args.script_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]
    if not has_output_dir_arg(forwarded_args):
        forwarded_args = ["--output-dir", str(DEFAULT_OUTPUT_DIR), *forwarded_args]

    vendor_package_root = str(VENDOR_ROOT)
    if vendor_package_root not in sys.path:
        sys.path.insert(0, vendor_package_root)

    sys.argv = [str(script_path), *forwarded_args]
    print(f"Launching {args.command} from {script_path.relative_to(REPO_ROOT)}")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
