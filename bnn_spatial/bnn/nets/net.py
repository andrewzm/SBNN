"""
Blank neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation_fns import *
from ..layers.layer import BlankLayer
from ..layers.embedding_layer import EmbeddingLayer


class BlankNet(nn.Module):
    def __init__(self, output_dim, hidden_dims, activation_fn, input_dim=None):
        """
        Neural network to be initialised for usage with SGHMC.

        :param output_dim: int, number of dimensions of network output
        :param hidden_dims: list, contains number of nodes for each hidden layer
        :param activation_fn: str, specify activation/nonlinearity used in network
        :param input_dim: (optional) int, number of dimensions of network input, if embedding layer is not used
        """
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim

        # Setup activation function
        options = {'cos': torch.cos, 'tanh': torch.tanh, 'relu': F.relu,
                   'softplus': F.softplus, 'rbf': rbf, 'linear': linear,
                   'sin': torch.sin, 'leaky_relu': F.leaky_relu, 'swish': swish}
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn

        self.layers = nn.ModuleList()

        # If embedding layer is not used
        if input_dim is not None:
            self.layers.add_module("hidden_0", BlankLayer(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.add_module("hidden_{}".format(i), BlankLayer(hidden_dims[i-1], hidden_dims[i]))

        # Output layer
        self.layers.add_module('output', BlankLayer(hidden_dims[-1], output_dim))

    def reset_parameters(self):
        """
        Reset parameters in each layer to values sampled from std normal distribution.
        """
        for m in self.layers.modules():
            if isinstance(m, BlankLayer):
                m.reset_parameters()

    def forward(self, X):
        """
        Performs forward pass through the whole network given input data X.

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :return: torch.Tensor, size (batch_size, output_dim), output data
        """
        for layer in list(self.layers)[:-1]:
            X = self.activation_fn(layer(X))

        output_layer = list(self.layers)[-1]
        X = output_layer(X)

        return X
