"""
Compatibility shim for the archived synthetic stack.

The implementation now lives under `experiments.legacy.synthetic`, but the old
import path remains available so archived scripts and docs still resolve.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "legacy" / "synthetic")]
