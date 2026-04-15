"""
GDFM package.

For now this package owns the extracted network modules plus the canonical
network-setup entrypoint. The main GDFM prototype still lives in
prototyping/gdfm_components.py while we stabilize the sampler side and the
package layout independently.
"""

from gdfm.model_setup import build_network
from gdfm.networks import BonnaireUNet, JiT, TimeCondMLP

__all__ = ["build_network", "BonnaireUNet", "JiT", "TimeCondMLP"]
