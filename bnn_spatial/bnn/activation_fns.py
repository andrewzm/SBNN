"""
Activation functions for neural networks
"""

__all__ = ['rbf', 'linear', 'sin', 'cos', 'swish']

import torch

# RBF function
def rbf(x):
    if len(x.shape) == 2:
        return torch.exp(-torch.norm(x, dim=1) ** 2)
    elif len(x.shape) == 1:
        return torch.exp(-torch.norm(x, dim=0) ** 2)
    else:
        raise IndexError('x has incorrect dimensions')

# RBF with length-scale (same as RBF above when l=1)
def rbf_scale(x, l):
    if len(x.shape) == 2:
        return torch.exp(-(torch.norm(x, dim=1) / l) ** 2)
    elif len(x.shape) == 1:
        return torch.exp(-(torch.norm(x, dim=0) / l) ** 2)
    else:
        raise IndexError('x has incorrect dimensions')

# Linear function
def linear(x):
    return x

# Sin function
def sin(x):
    return torch.sin(x)

# Cos function
def cos(x):
    return torch.cos(x)

# Swish function
def swish(x):
    return x * torch.sigmoid(x)
