"""
Spatial embedding layer containing RBF units
"""

import numpy as np
import torch
import torch.nn as nn
from ..activation_fns import rbf, rbf_scale


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, domain, rbf_ls=1):
        """
        Implementation of embedding layer for BNN input.

        :param input_dim: int, number of dimensions of previous layer's output (this layer's input)
        :param output_dim: int, number of dimensions of this layer's output
        :param domain: torch.Tensor, contains all test inputs in the rows
        :param rbf_ls: float, length-scale of spatial basis functions (RBFs)
        """
        super(EmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # must be a perfect square for input_dim=2
        self.domain = domain
        self.rbf_ls = rbf_ls
  
    def forward(self, X):
        """
        Performs forward pass through layer given input data.

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :return: torch.Tensor, size (batch_size, output_dim), output data
        """
        if self.input_dim == 2:
            x1_min, x1_max = self.domain[:, 0].min().item(), self.domain[:, 0].max().item()
            x2_min, x2_max = self.domain[:, 1].min().item(), self.domain[:, 1].max().item()
            batch_size = X.shape[0]
            X1_subset = torch.linspace(x1_min, x1_max, int(np.sqrt(self.output_dim)))
            X2_subset = torch.linspace(x2_min, x2_max, int(np.sqrt(self.output_dim)))
            X1_coords, X2_coords = torch.meshgrid(X1_subset, X2_subset)
            test_subset = torch.vstack((X1_coords.flatten(), X2_coords.flatten())).T.to(X.device).float()
            output = torch.zeros_like(test_subset[:, 0])  # size (output_dim)
        elif self.input_dim == 1:
            x_min, x_max = self.domain.min().item(), self.domain.max().item()
            batch_size = X.shape[0]
            test_subset = torch.linspace(x_min, x_max, self.output_dim)
            output = torch.zeros_like(test_subset)
        else:
            raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

        # Rows of 'output' correspond to points within input batch; columns to radial basis functions
        output = torch.unsqueeze(output, 0).repeat([batch_size, 1])  # size (batch_size, output_dim = # RBFs)
        for i, x0 in enumerate(test_subset):  # iterates through test_subset row-by-row
            output[:, i] = rbf_scale(torch.subtract(X, x0), l=self.rbf_ls).squeeze()
        return output
