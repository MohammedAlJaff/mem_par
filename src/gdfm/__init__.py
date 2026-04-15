"""
GDFM package.

This package is the canonical home for the first-party GDFM model core plus
the extracted network modules and network-setup entrypoint.
"""

from gdfm.core import GDFM, print_forward_shapes, print_model_summary
from gdfm.model_setup import build_network
from gdfm.networks import BonnaireUNet, JiT, TimeCondMLP

__all__ = [
    "BonnaireUNet",
    "GDFM",
    "JiT",
    "TimeCondMLP",
    "build_network",
    "print_forward_shapes",
    "print_model_summary",
]
