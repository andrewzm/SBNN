"""
Stochastic layer for GPi-G BNN prior parameterisation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def softplus_inv(x):
    return math.log(math.exp(x) - 1)

class GaussianLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rbf_dim=None, prior_per='layer', fit_means=True, 
                       nonstationary=False, init_std=None):
        """
        Implementation of BNN prior layer with Gaussian prior over parameters.

        :param input_dim: int, number of dimensions of previous layer's output (this layer's input)
        :param output_dim: int, number of dimensions of this layer's output
        :param rbf_dim: int, width of embedding layer (number of spatial basis functions)
        :param prior_per: str, indicates either one prior per `layer` or `parameter`
        :param fit_means: bool, specify if means are fitted as parameters (set to zero otherwise)
        :param nonstationary: bool, specify if spatial dependence is incorporated into hyperparameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonstationary = nonstationary
        self.prior_per = prior_per
        self.fit_means = fit_means

        if init_std is not None:
            self.init_rho = softplus_inv(init_std)
        else:
            self.init_rho = 1

        if nonstationary and rbf_dim is None:
            raise Exception('Must specify number of RBFs for nonstationary case.')

        # Set dimensions of optimised parameters
        if prior_per == 'layer':
            if nonstationary:
                self.W_shape, self.b_shape = (rbf_dim, 1), (rbf_dim, 1)
            else:
                self.W_shape, self.b_shape = 1, 1
        elif prior_per == 'parameter':
            if nonstationary:
                self.W_shape, self.b_shape = (rbf_dim, input_dim, output_dim), (rbf_dim, output_dim)
            else:
                self.W_shape, self.b_shape = (input_dim, output_dim), output_dim
        else:
            raise ValueError("Accepted values for prior_per: `parameter` or `layer`")

        # Define optimised hyperparameters and require gradient (make autograd record operations)
        if self.nonstationary:
            if fit_means:
                self.W_mu_coeffs = nn.Parameter(torch.zeros(self.W_shape), requires_grad=True)
                self.b_mu_coeffs = nn.Parameter(torch.zeros(self.b_shape), requires_grad=True)
            self.W_rho_coeffs = nn.Parameter(self.init_rho * torch.randn(self.W_shape), requires_grad=True)
            self.b_rho_coeffs = nn.Parameter(self.init_rho * torch.randn(self.b_shape), requires_grad=True)
        else:
            if fit_means:
                self.W_mu = nn.Parameter(torch.zeros(self.W_shape), requires_grad=True)
                self.b_mu = nn.Parameter(torch.zeros(self.b_shape), requires_grad=True)
            self.W_rho = nn.Parameter(self.init_rho * torch.ones(self.W_shape), requires_grad=True)
            self.b_rho = nn.Parameter(self.init_rho * torch.ones(self.b_shape), requires_grad=True)

    def reset_parameters(self):
        """
        Reset parameters to values sampled from std normal distribution.
        """
        if self.nonstationary:
            init.normal_(self.init_rho * self.W_rho_coeffs)
            init.normal_(self.init_rho * self.b_rho_coeffs)
            if self.fit_means:
                init.zeros_(self.W_mu_coeffs)
                init.zeros_(self.b_mu_coeffs)
        else:
            init.ones_(self.init_rho * self.W_rho)
            init.ones_(self.init_rho * self.b_rho)
            if self.fit_means:
                init.zeros_(self.W_mu)
                init.zeros_(self.b_mu)

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
            W_std = F.softplus(torch.tensordot(X_RBF, self.W_rho_coeffs, dims=([1],[0])))  # [batch_size, W_shape]
            b_std = F.softplus(torch.tensordot(X_RBF, self.b_rho_coeffs, dims=([1],[0])))  # [batch_size, b_shape]
            if self.fit_means:
                W_mu = torch.tensordot(X_RBF, self.W_mu_coeffs, dims=([1],[0]))  # [batch_size, W_shape]
                b_mu = torch.tensordot(X_RBF, self.b_mu_coeffs, dims=([1],[0]))  # [batch_size, b_shape]
            else:
                W_mu = 0.
                b_mu = 0.

            # Adjust for the case when W_mu and W_std have shape [batch_size, 1]
            if (len(W_std.shape) == 2) & (W_std.shape[-1] == 1):
                W_std = W_std.unsqueeze(1)
                if isinstance(W_mu, torch.Tensor):
                    W_mu = W_mu.unsqueeze(1)

            W = W_mu + W_std * torch.randn((1, self.input_dim, self.output_dim), device=W_std.device)
            W = W / math.sqrt(self.input_dim)  # NTK
            b = b_mu + b_std * torch.randn((1, self.output_dim), device=b_std.device)

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

            if self.prior_per == 'layer':
                W_std = F.softplus(X_RBF @ self.W_rho_coeffs).squeeze()[None, :, None, None]
                b_std = F.softplus(X_RBF @ self.b_rho_coeffs).squeeze()[None, :, None]
            elif self.prior_per == 'parameter':
                W_std = F.softplus(torch.tensordot(X_RBF, self.W_rho_coeffs, dims=([1],[0]))).unsqueeze(0)
                b_std = F.softplus(torch.tensordot(X_RBF, self.b_rho_coeffs, dims=([1],[0]))).unsqueeze(0)

            if self.fit_means:
                if self.prior_per == 'layer':
                    W_mu = (X_RBF @ self.W_mu_coeffs).squeeze()[None, :, None, None]
                    b_mu = (X_RBF @ self.b_mu_coeffs).squeeze()[None, :, None]
                elif self.prior_per == 'parameter':
                    W_mu = torch.tensordot(X_RBF, self.W_mu_coeffs, dims=([1],[0])).unsqueeze(0)
                    b_mu = torch.tensordot(X_RBF, self.b_mu_coeffs, dims=([1],[0])).unsqueeze(0)
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
