"""
Network modules used by GDFM-family models.

The MLP is the canonical prototype backbone. The UNet and JiT modules are
vendor-aligned adapters so model setup can target the same architectural
surfaces the final experiments need, without pulling training or GDFM logic
into this package.
"""

from gdfm.networks.jit import JiT
from gdfm.networks.mlp import TimeCondMLP
from gdfm.networks.unet import BonnaireUNet

__all__ = ["BonnaireUNet", "JiT", "TimeCondMLP"]
