import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
from typing import Literal


"""
A dfm model is specified by the choice of functional forms of a(t), b(t), and c(t) and d(t).
a(t) and b(t) determine the noise/latent interpolation schedule.
c(t) and d(t) determine the regression target.

x_t = a(t) * x_0 + b(t) * z
where z ~ N(0, I)

and regression target is:
r(x, t, z) = c(t) * x_t + d(t) * z

The model F learns to predict r(x, t, z)

loss(F) = ||F(x_t, t) - r(x, t, z)||^2

Parameterization refers to two things: 
- what the direct output of the model F represents
- what the regression target r(x, t, z) represents

"""


def dfm_settings():
    """
    Sets up the settings for the diffusion/flow matching model.
    """
    pass


class DFM(nn.Module):
    """
    Diffusion/Flow Matching model.
    """

    def __init__(self, network, args):
        super().__init__()
        self.network = network

    def __setup_abcd(self):
        # we need to set up a(t), b(t), c(t), d(t)
        # either directly or by specifying the functional forms
        # or by the model family mentioned in the args, 
        # eg 
        #   if args.model_family == "ddpm",
        #       a(t) = 1, 
        #       b(t) = sqrt(1 - exp(-beta_t)), 
        #       c(t) = 0, 
        #       d(t) = sqrt(exp(-beta_t))
        #  if args.model_family == "edm",
        #       a(t) = exp(-beta_t/2), 
        #       b(t) = sqrt(1 - exp(-beta_t)), 
        #       c(t) = 0, 
        #       d(t) = sqrt(exp(-beta_t))
        #  if args.model_family == "fm",
        #       a(t) = 0, 
        #       b(t) = 1, 
        #       c(t) = 0, 
        #       d(t) = 1
        pass


    def generate_sample(self, N, t):
        z = torch.randn(N, *self.network.input_shape)
        return self.network(z, t)

    def forward(self, x, t):
        return self.network(x, t)


if __name__ == "__main__":
    ################
    #### Test DFM ####
    ################

    """
    What this does:
    - Sets up a simple general diffusion/flow matching model.
    - verify that we can get DDPM, EDM and FM models by specifying different functional params

    What to expect:
    """
    pass
