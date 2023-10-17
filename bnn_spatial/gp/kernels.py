"""
Gaussian process kernel functions
"""

import torch
import numpy as np
import torch.linalg as la

# Ran into problems using scipy's distance_matrix on GPU (25/08)

def RBF(dist, ampl, leng, power=2):
    """
    Radial basis function kernel, aka power exponential kernel (SE is a special case).

    :param ampl: float, amplitude parameter
    :param leng: float, lengthscale parameter
    :param power: float, with 0 < power <= 2, specifying exponent in power-exponential kernel (SE has power = 2)
    """
    # Enforce restriction on power values
    if power <= 0 or power > 2:
        raise Exception('Require values 0 < power <= 2 for power exponential kernel.')

    # Adjust lengthscale for squared exponential case
    if power == 2:
        leng *= np.sqrt(2)

    return ampl * torch.exp(-1 * (dist / leng) ** power)

def Matern32(dist, ampl, leng):
    """
    Matern 3/2 kernel.

    :param ampl: float, amplitude parameter
    :param leng: float, lengthscale parameter
    """
    return ampl * (1. + np.sqrt(3.) * dist / leng) * torch.exp(-np.sqrt(3.) * dist / leng)

