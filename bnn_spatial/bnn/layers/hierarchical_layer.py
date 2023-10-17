"""
Stochastic layer for GPi-H BNN prior parameterisation
"""

import math
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class HierarchicalLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rbf_dim=None, prior_per='layer', fit_means=True):
        """
        Implementation of each BNN layer, when using fixed Gaussian prior over parameters.

        :param input_dim: int, number of dimensions of previous layer's output (this layer's input)
        :param output_dim: int, number of dimensions of this layer's output
        :param rbf_dim: int, width of embedding layer (number of spatial basis functions)
        :param prior_per: str, indicates either one prior per `layer`, `parameter`, or `input`
        :param fit_means: bool, specify if means are fitted as parameters (set to zero otherwise)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonstationary = False
        self.fit_means = fit_means

        # Set dimensions of optimised parameters
        if prior_per == 'layer':
            W_shape, b_shape = 1, 1
        elif prior_per == 'parameter':
            W_shape, b_shape = (input_dim, output_dim), output_dim
        elif prior_per == 'input':
            if rbf_dim is None:
                raise Exception('For nonstationary prior (one per input), require rbf_dim (embedding layer width)')
            W_shape, b_shape = (rbf_dim, 1), (rbf_dim, 1)
            self.nonstationary = True
        else:
            raise ValueError("Accepted values: `parameter`, `layer`, or `input`")

        # Define optimised hyperparameters and require gradient (make autograd record operations)
        if self.nonstationary:
            if fit_means:
                self.W_mu_coeffs = nn.Parameter(torch.zeros(W_shape), requires_grad=True)
                self.b_mu_coeffs = nn.Parameter(torch.zeros(b_shape), requires_grad=True)
            self.W_shape_coeffs = nn.Parameter(torch.randn(W_shape), requires_grad=True)
            self.W_rate_coeffs = nn.Parameter(torch.randn(W_shape), requires_grad=True)
            self.b_shape_coeffs = nn.Parameter(torch.randn(b_shape), requires_grad=True)
            self.b_rate_coeffs = nn.Parameter(torch.randn(b_shape), requires_grad=True)
        else:
            if fit_means:
                self.W_mu = nn.Parameter(torch.zeros(W_shape), requires_grad=True)
                self.b_mu = nn.Parameter(torch.zeros(b_shape), requires_grad=True)
            self.W_shape = nn.Parameter(torch.randn(W_shape), requires_grad=True)
            self.W_rate = nn.Parameter(torch.randn(W_shape), requires_grad=True)
            self.b_shape = nn.Parameter(torch.randn(b_shape), requires_grad=True)
            self.b_rate = nn.Parameter(torch.randn(b_shape), requires_grad=True)

    def reset_parameters(self):
        """
        Reset parameters to values sampled from std normal distribution.
        """
        if self.nonstationary:
            init.normal_(self.W_shape_coeffs)
            init.normal_(self.W_rate_coeffs)
            init.normal_(self.b_shape_coeffs)
            init.normal_(self.b_rate_coeffs)
            if self.fit_means:
                init.zeros_(self.W_mu_coeffs)
                init.zeros_(self.b_mu_coeffs)
        else:
            init.normal_(self.W_shape)
            init.normal_(self.W_rate)
            init.normal_(self.b_shape)
            init.normal_(self.b_rate)
            if self.fit_means:
                init.zeros_(self.W_mu)
                init.zeros_(self.b_mu)

    def _resample_std(self, X_RBF):
        """
        Obtain std deviations from resampled inverse-gamma variances.

        :param X_RBF: torch.Tensor, embedding layer output (for nonstationary case)
        :return: tuple (torch.Tensor, torch.Tensor), weight std dev, bias std dev
        """
        # Positivity constraints
        if self.nonstationary:
            W_shape = F.softplus(X_RBF @ self.W_shape_coeffs).squeeze()
            W_rate = F.softplus(X_RBF @ self.W_rate_coeffs).squeeze()
            b_shape = F.softplus(X_RBF @ self.b_shape_coeffs).squeeze()
            b_rate = F.softplus(X_RBF @ self.b_rate_coeffs).squeeze()
        else:
            W_shape = F.softplus(self.W_shape)
            W_rate = F.softplus(self.W_rate)
            b_shape = F.softplus(self.b_shape)
            b_rate = F.softplus(self.b_rate)

        # Resample variances (sample from Gamma then invert)
        W_gamma_dist = dist.Gamma(W_shape, W_rate)
        b_gamma_dist = dist.Gamma(b_shape, b_rate)

        # Note: rsample() is reparametrised sample, which stores gradients
        inv_W_var = W_gamma_dist.rsample()
        inv_b_var = b_gamma_dist.rsample()

        # Note: self.eps added in denominator to avoid division by zero
        W_std = 1. / (torch.sqrt(inv_W_var) + 1e-10)
        b_std = 1. / (torch.sqrt(inv_b_var) + 1e-10)

        return W_std, b_std

    def forward(self, X, X_RBF=None):
        """
        Performs forward pass through layer given input data.

        :param X: torch.Tensor, size (batch_size, input_dim), input data
        :return: torch.Tensor, size (batch_size, output_dim), output data
        """
        if self.nonstationary:
            X = X.to(self.W_rho_coeffs.device)
            if X_RBF is None:
                X_RBF = X.detach().clone()
            else:
                X_RBF = X_RBF.to(self.W_rho_coeffs.device)
            W_std, b_std = self._resample_std(X_RBF)  # b_std has shape [batch_size, 1]
            W_std = W_std.unsqueeze(2)  # need shape [batch_size, 1, 1]
            if self.fit_means:
                W_mu = (X_RBF @ self.W_mu_coeffs).unsqueeze(2)  # [batch_size, 1, 1]
                b_mu = (X_RBF @ self.b_mu_coeffs)  # [batch_size, 1]
            else:
                W_mu = 0.
                b_mu = 0.
            Zw = torch.randn((1, self.input_dim, self.output_dim), device=W_std.device)
            Zb = torch.randn((1, self.output_dim), device=b_std.device)
            W = W_mu + W_std * Zw
            W = W / math.sqrt(self.input_dim)  # NTK
            b = b_mu + b_std * Zb

            # Rearrange tensors for broadcasting to work properly
            if len(X.shape) < len(W.shape):
                X = X.unsqueeze(1)
            fwd = (X @ W).squeeze() + b.squeeze()
            # X @ W is [batch_size, 1, input_dim] * [batch_size, input_dim, output_dim] = [batch_size, 1, output_dim]
            return fwd
        else:
            X = X.to(self.W_rho.device)
            if self.fit_means:
                W_mu = self.W_mu
                b_mu = self.b_mu
            else:
                W_mu = 0.
                b_mu = 0.
            W = W_mu + F.softplus(self.W_rho) * torch.randn((self.input_dim, self.output_dim), device=self.W_rho.device)
            b = b_mu + F.softplus(self.b_rho) * torch.randn(self.output_dim, device=self.b_rho.device)
            W = W / math.sqrt(self.input_dim)  # NTK
            return X @ W + b

    def sample_predict(self, X, n_samples, X_RBF=None):
        """
        Perform predictions using n_samples different sampled network parameters.

        :param X: torch.Tensor, size (batch_size, input_dim) or (n_samples, batch_size, input_dim), input data
        :param n_samples: int, number of network samples
        :param X_RBF: torch.Tensor, contains embedding layer output (RBF values in each neuron)
        :return: torch.Tensor, size (n_samples, batch_size, output_dim), output data
        """
        if self.nonstationary:
            X = X.to(self.W_rho_coeffs.device)

            # Resize input X appropriately
            if len(X.shape) == 2:
                X_RBF = X.detach().clone()
                X = X[None, :, None, :].repeat(n_samples, 1, 1, 1)
            else:
                X_RBF = X_RBF.to(self.W_rho_coeffs.device)
                X = X.unsqueeze(2)

            W_std, b_std = self._resample_std(X_RBF)
            W_std = W_std.squeeze()[None, :, None, None]
            b_std = b_std.squeeze()[None, :, None]
            if self.fit_means:
                W_mu = (X_RBF @ self.W_mu_coeffs).squeeze()[None, :, None, None]
                b_mu = (X_RBF @ self.b_mu_coeffs).squeeze()[None, :, None]
            else:
                W_mu = 0.
                b_mu = 0.
            Ws = W_mu + W_std * torch.randn((n_samples, 1, self.input_dim, self.output_dim), device=W_std.device)
            bs = b_mu + b_std * torch.randn((n_samples, 1, self.output_dim), device=b_std.device)
            Ws = Ws / math.sqrt(self.input_dim)  # NTK
            return (X @ Ws).squeeze() + bs.squeeze()
        else:
            X = X.to(self.W_rho.device)
            if self.fit_means:
                W_mu = self.W_mu
                b_mu = self.b_mu
            else:
                W_mu = 0.
                b_mu = 0.
            Ws = W_mu + F.softplus(self.W_rho) * torch.randn((n_samples, self.input_dim, self.output_dim),
                                                             device=self.W_rho.device)
            bs = b_mu + F.softplus(self.b_rho) * torch.randn((n_samples, 1, self.output_dim),
                                                             device=self.b_rho.device)
            Ws = Ws / math.sqrt(self.input_dim)  # NTK
            return X @ Ws + bs
