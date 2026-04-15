"""
Compatibility shim for older imports.

Canonical code now lives in `gdfm.core`, but `gdfm.model` remains importable so
the transition does not break existing callers.
"""

from gdfm.core import GDFM, print_forward_shapes, print_model_summary

__all__ = ["GDFM", "print_forward_shapes", "print_model_summary"]
