"""
BNN with GPi-G prior parameterisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from ..activation_fns import *
from ..layers.gaussian_layer import GaussianLayer
from ..layers.embedding_layer import EmbeddingLayer


class GaussianNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation_fn, domain=None, prior_per='layer',
                 fit_means=False, rbf_ls=1, nonstationary=False, init_std=None):
        """
        Implementation of BNN prior with Gaussian prior over parameters.

        :param input_dim: int, number of dimensions of network input
        :param output_dim: int, number of dimensions of network output
        :param hidden_dims: list, contains number of nodes for each hidden layer
        :param activation_fn: str, specify activation/nonlinearity used in network
        :param domain: torch.Tensor, contains all X1 and X2 input coordinates in the first and second cols
        :param prior_per: str, indicates either one prior per `layer`, `parameter`, or `input`
        :param fit_means: bool, specify if means are fitted as parameters (set to zero otherwise)
        :param rbf_ls: float, lengthscale for embedding layer RBFs
        :param nonstationary: bool, specify if spatial dependence is incorporated into hyperparameters
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
        self.layers = nn.ModuleList()
        if domain is None:
            self.layers.add_module('hidden_0', GaussianLayer(input_dim=input_dim,
                                                             output_dim=hidden_dims[0],
                                                             prior_per=prior_per,
                                                             fit_means=fit_means,
                                                             nonstationary=nonstationary,
                                                             init_std=init_std))
            rbf_dim = None
        else:
            rbf_dim = hidden_dims[0]
            if (int(np.sqrt(rbf_dim)) ** 2 != rbf_dim) and (input_dim == 2):
                raise Exception('For embedding layer, require the first hidden dim to be a perfect square')
            self.layers.add_module('embedding', EmbeddingLayer(input_dim=input_dim,
                                                               output_dim=rbf_dim,
                                                               domain=domain,
                                                               rbf_ls=rbf_ls))

        # Note: nn.ModuleList holds network submodules in a list, and submodules can be added with add_module.
        #       Each submodule (layer) in turn contains the parameters W_rho and b_rho which are optimised.

        # Initialise the hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.add_module('hidden_{}'.format(i), GaussianLayer(input_dim=hidden_dims[i-1],
                                                                        output_dim=hidden_dims[i],
                                                                        rbf_dim=rbf_dim,
                                                                        prior_per=prior_per,
                                                                        fit_means=fit_means,
                                                                        nonstationary=nonstationary,
                                                                        init_std=init_std))

        # Initialise output layer
        self.layers.add_module('output', GaussianLayer(input_dim=hidden_dims[-1],
                                                       output_dim=output_dim,
                                                       rbf_dim=rbf_dim,
                                                       prior_per=prior_per,
                                                       fit_means=fit_means,
                                                       nonstationary=nonstationary,
                                                       init_std=init_std))

    def reset_parameters(self):
        """
        Reset optimised hyperparameters in each non-deterministic layer.
        """
        for m in self.layers.modules():
            if isinstance(m, GaussianLayer):
                m.reset_parameters()

    def forward(self, X):
        """
        Performs forward pass through the whole network given input data X.

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :return: torch.Tensor, size (batch_size, output_dim), output data
        """
        named_layers = self.layers.named_modules()

        # Propagate input through each layer
        X_RBF = None
        for name, layer in list(named_layers):
            if 'embedding' in name:
                X = layer(X)
                X_RBF = deepcopy(X)
            elif 'hidden' in name:
                X = self.activation_fn(layer(X, X_RBF))
            elif 'output' in name:
                X = layer(X, X_RBF)

        return X

    def sample_functions(self, X, n_samples):
        """
        Performs predictions with BNN at points X, for n_samples different parameter samples (i.e. different BNNs).

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :param n_samples: int, number of network samples
        :return: torch.Tensor, size (batch_size, n_samples, output_dim), output data
        """
        named_layers = self.layers.named_modules()

        # Propagate input through each layer
        X_RBF = None
        for name, layer in list(named_layers):
            if 'embedding' in name:
                X = layer(X)
                X_RBF = deepcopy(X)
            if 'hidden' in name:
                X = self.activation_fn(layer.sample_predict(X, n_samples, X_RBF))
            elif 'output' in name:
                X = layer.sample_predict(X, n_samples, X_RBF)

        # Return network output, with size (batch_size, n_samples, output_dim) after resizing
        X = torch.transpose(X, 0, 1)  # need to rearrange in this manner for compatibility with wasserstein_mapper.py
        return X

    def network_parameters(self):
        """
        Obtain std dev values for all weights and biases throughout the network (stationary case only).

        :return: tuple of lists, each containing std devs (of weights and biases) for each layer
        """
        W_std_list = []
        b_std_list = []
        for name in self.state_dict().keys():
            param = self.state_dict()[name]
            if '.W' in name:
                W_std = round(float(F.softplus(param)), 6)
                W_std_list.append(W_std)
            elif '.b' in name:
                b_std = round(float(F.softplus(param)), 6)
                b_std_list.append(b_std)

        return W_std_list, b_std_list
