"""
Compatibility shim for the archived natural-images stack.

The implementation now lives under `experiments.legacy.natural_images`, but the
old import path remains available so archived scripts and docs still resolve.
"""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent.parent / "legacy" / "natural_images")
]
