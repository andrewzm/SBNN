"""
Likelihood modules
"""

import torch
import torch.nn as nn


class LikGaussian(nn.Module):
    def __init__(self, var):
        """
        Gaussian likelihood module.

        :param var: float, measurement error variance (sn2)
        """
        super().__init__()
        self.var = var

        # Note: taking reduction='mean' in MSELoss gives MSE, while reduction='sum' gives RSS

    def forward(self, fx, y):
        """
        Forward pass through Gaussian likelihood module, returning negative log likelihood.

        :param fx: torch.Tensor, network predictions
        :param y: torch.Tensor, corresponding noisy targets
        :return: float, negative log likelihood
        """
        return -self.loglik(fx, y)  # output negative log likelihood when LikGaussian is called

    def loglik(self, fx, y):
        """
        Compute log likelihood.

        :param fx: torch.Tensor, network predictions
        :param y: torch.Tensor, corresponding noisy targets
        :return: float, Gaussian log likelihood
        """
        return - 0.5 * (torch.sum((fx - y) ** 2) / self.var)
