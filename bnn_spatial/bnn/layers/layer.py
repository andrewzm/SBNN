"""
Blank layer with weights and biases stored as parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BlankLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Standard hidden layer for BNN (parameters overridden by values imported from checkpoint when optimising).

        :param input_dim: int, size of layer input
        :param output_dim: int, size of layer output
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        W_shape, b_shape = (input_dim, output_dim), output_dim

        # Initialize the parameters
        self.W = nn.Parameter(torch.randn(W_shape), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(b_shape), requires_grad=True)

    def reset_parameters(self):
        """
        Reset parameters to values sampled from std normal distribution.
        """
        init.normal_(self.W)
        init.zeros_(self.b)

    def forward(self, X):
        """
        Performs forward pass through layer given input data.

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :return: torch.tensor, size (batch_size, output_dim), output data
        """
        W = self.W
        W = W / math.sqrt(self.input_dim)  # NTK parametrisation
        b = self.b
        return X @ W + b
