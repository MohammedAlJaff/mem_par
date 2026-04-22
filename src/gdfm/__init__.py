"""
GDFM package.

This package is the canonical home for the first-party GDFM model core, the
network modules, the network factory, and standalone sampler utilities.
"""

from gdfm.core import GDFM, print_forward_shapes, print_model_summary
from gdfm.model_setup import build_network
from gdfm.networks import BonnaireUNet, JiT, TimeCondMLP
from gdfm.samplers import (
    convert_prediction,
    model_to_eps,
    model_to_score,
    model_to_velocity,
    model_to_x,
    predict_in_prediction_space,
    sample,
    sample_euler,
    sample_heun,
    sample_vp_reverse_sde,
    time_grid,
    vp_beta,
    vp_probability_flow_drift,
    vp_reverse_sde_diffusion,
    vp_reverse_sde_drift,
)

__all__ = [
    "BonnaireUNet",
    "GDFM",
    "JiT",
    "TimeCondMLP",
    "build_network",
    "convert_prediction",
    "model_to_eps",
    "model_to_score",
    "model_to_velocity",
    "model_to_x",
    "predict_in_prediction_space",
    "sample",
    "sample_euler",
    "sample_heun",
    "sample_vp_reverse_sde",
    "print_forward_shapes",
    "print_model_summary",
    "time_grid",
    "vp_beta",
    "vp_probability_flow_drift",
    "vp_reverse_sde_diffusion",
    "vp_reverse_sde_drift",
]
