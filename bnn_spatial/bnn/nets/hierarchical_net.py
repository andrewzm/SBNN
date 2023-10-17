"""
BNN with GPi-H prior parameterisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from ..activation_fns import *
from ..layers.hierarchical_layer import HierarchicalLayer
from ..layers.embedding_layer import EmbeddingLayer


class HierarchicalNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation_fn,
                 domain=None, prior_per='layer', fit_means=False, rbf_ls=1):
        """
        Implementation of BNN prior with fixed Gaussian prior over parameters.

        :param input_dim: int, number of dimensions of network input
        :param output_dim: int, number of dimensions of network output
        :param hidden_dims: list, contains number of nodes for each hidden layer
        :param activation_fn: str, specify activation/nonlinearity used in network
        :param domain: torch.Tensor, contains all X1 and X2 input coordinates in the first and second cols
        :param prior_per: str, indicates either one prior per `layer`, `parameter`, or `input`
        :param fit_means: bool, specify if means are fitted as parameters (set to zero otherwise)
        :param rbf_ls: float, lengthscale for embedding layer RBFs
        """
        super().__init__()
        self.input_dim = input_dim

        # Set up activation function
        options = {'cos': torch.cos, 'tanh': torch.tanh, 'relu': F.relu,
                   'softplus': F.softplus, 'rbf': rbf, 'linear': linear,
                   'sin': sin, 'leaky_relu': F.leaky_relu, 'swish': swish}
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn

        # Initialise layers: self.layers stores hidden layers in ModuleList
        if domain is None:
            self.layers = nn.ModuleList()
            self.layers.add_module('hidden_0', HierarchicalLayer(input_dim=input_dim,
                                                                 output_dim=hidden_dims[0],
                                                                 prior_per=prior_per,
                                                                 fit_means=fit_means))
            rbf_dim = None
        else:
            rbf_dim = hidden_dims[0]
            if (int(np.sqrt(rbf_dim)) ** 2 != rbf_dim) and (input_dim == 2):
                raise Exception('For embedding layer, require the first hidden dim to be a perfect square')
            self.layers = nn.ModuleList([EmbeddingLayer(input_dim, rbf_dim, domain, rbf_ls=rbf_ls)])

        # Note: nn.ModuleList holds network submodules in a list, and submodules can be added with add_module.
        #       Each submodule (layer) in turn contains the parameters W_rho and b_rho which are optimised.

        # Initialise the hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.add_module('hidden_{}'.format(i), HierarchicalLayer(input_dim=hidden_dims[i - 1],
                                                                            output_dim=hidden_dims[i],
                                                                            rbf_dim=rbf_dim,
                                                                            prior_per=prior_per,
                                                                            fit_means=fit_means))

        # Initialise output layer
        self.output_layer = HierarchicalLayer(input_dim=hidden_dims[-1],
                                              output_dim=output_dim,
                                              rbf_dim=rbf_dim,
                                              prior_per=prior_per,
                                              fit_means=fit_means)

    def reset_parameters(self):
        """
        Reset optimised hyperparameters in each non-deterministic layer.
        """
        for m in self.modules():
            if isinstance(m, HierarchicalLayer):
                m.reset_parameters()

    def forward(self, X):
        """
        Performs forward pass through the whole network given input data X.

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :return: torch.Tensor, size (batch_size, output_dim), output data
        """
        # Apply RBFs to network input
        embedding_layer = list(self.layers)[0]
        X = embedding_layer(X)
        X_RBF = deepcopy(X)

        # Propagate input through hidden layers, applying activations
        for layer in list(self.layers)[1:]:
            X = self.activation_fn(layer(X, X_RBF))

        # Return network output (from output layer)
        X = self.output_layer(X, X_RBF)
        return X

    def sample_functions(self, X, n_samples):
        """
        Performs predictions with BNN at points X, for n_samples different parameter samples (i.e. different BNNs).

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :param n_samples: int, number of network samples
        :return: torch.Tensor, size (batch_size, n_samples, output_dim), output data
        """
        # Apply RBFs to network input
        embedding_layer = list(self.layers)[0]
        X = embedding_layer(X)
        X_RBF = deepcopy(X)

        # Propagate input through hidden layers, applying activations
        for layer in list(self.layers)[1:]:
            X = self.activation_fn(layer.sample_predict(X, n_samples, X_RBF))

        # Return network output, with size (batch_size, n_samples, output_dim) after resizing
        X = self.output_layer.sample_predict(X, n_samples, X_RBF)
        X = torch.transpose(X, 0, 1)  # need to rearrange in this manner for compatibility with wasserstein_mapper.py
        return X

    def network_parameters(self):
        """
        Obtain shape/rate hyperparameters values for all weights and biases throughout the network.

        :return: tuple of lists, each containing hyperprior shape and rate values for each layer
        """
        shape_list = list(self.state_dict().values())[0::2]  # even indices contain shape values for each layer
        rate_list = list(self.state_dict().values())[1::2]  # odd indices contain rate values for each layer
        W_shape_list = shape_list[0::2]  # contains shape values for weights
        b_shape_list = shape_list[1::2]  # contains shape values for biases
        W_rate_list = rate_list[0::2]  # contains rate values for weights
        b_rate_list = rate_list[1::2]  # contains rate values for biases

        # Apply softplus to all list entries, then convert to float and round
        W_shape_list = [round(float(F.softplus(ws)), 6) for ws in W_shape_list]
        b_shape_list = [round(float(F.softplus(bs)), 6) for bs in b_shape_list]
        W_rate_list = [round(float(F.softplus(wr)), 6) for wr in W_rate_list]
        b_rate_list = [round(float(F.softplus(br)), 6) for br in b_rate_list]

        return W_shape_list, b_shape_list, W_rate_list, b_rate_list
