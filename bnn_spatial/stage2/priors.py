"""
Prior modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class PriorModule(nn.Module):
    def __init__(self):
        """
        Parent class for the prior module.
        """
        super(PriorModule, self).__init__()
        self.hyperprior = False

    def forward(self, net, test_input=None):
        """
        Compute negative log joint prior.

        :param net: nn.Module, the input network to be evaluated
        :return: torch.Tensor, negative log joint prior
        """
        return -self.logp(net, test_input)

    def logp(self, net, test_input=None):
        """
        Compute log joint prior (implemented by child classes).

        :param net: nn.Module, the input network to be evaluated
        :return: torch.Tensor, log joint prior
        """
        raise NotImplementedError

"""
Gaussian Prior (Fixed and GPi-G)
"""

class FixedGaussianPrior(PriorModule):
    def __init__(self, mu=0.0, std=1.0):
        """
        Child class for fixed Gaussian prior over the parameters.

        :param mu: float, mean for all parameters
        :param std: float, std dev for all parameters
        """
        super(FixedGaussianPrior, self).__init__()
        self.mu = mu
        self.std = torch.Tensor([std])

    def logp(self, net, test_input=None):
        """
        Compute log joint prior.

        :param net: nn.Module, the input network to be evaluated
        :return: torch.Tensor, log joint prior
        """
        res = 0.
        for name, param in net.named_parameters():  # each param is a tensor of weights/biases
            if 'batch_norm' in name:
                continue
            var = self.std ** 2
            res -= 0.5 * torch.sum((param - self.mu) ** 2) / var
        return res

class OptimGaussianPrior(PriorModule):
    def __init__(self, saved_path, rbf=None, device="cpu"):
        """
        Child class for optimised Gaussian prior over the parameters (GPi-G).

        :param saved_path: str, path to checkpoint containing optimised parameters
        :param rbf: torch.Tensor, embedding layer evaluations on all spatial inputs
        :param device: str, specify device for module
        """
        super(OptimGaussianPrior, self).__init__()
        self.params = {}
        self.device = device

        # Embedding layer RBF evaluations
        self.rbf = rbf

        data = torch.load(saved_path, map_location=torch.device(self.device))
        for name, param in data.items():  # hyperparam tensors contain rho and mu coefficients (for weights and biases)
            self.params[name] = param.squeeze().to(self.device)

    def to(self, device):
        """
        Move each network parameter to the configured device.

        :param device: str, specify device to transfer to
        :return: instance of OptimGaussianPrior
        """
        for name in self.params.keys():
            self.params[name] = self.params[name].to(device)
            if self.rbf is not None:
                self.rbf = self.rbf.to(device)
        return self

    def _get_params_by_name(self, name, test_input=None):
        """
        Extract hyperparameters for layer by specifying name of corresponding parameters.

        :param name: str, name of parameters
        :param test_input: int, specifies row index of test input
        :return: tuple, 2*(float) or 2*(torch.Tensor), mean and std dev for the layer's parameters
        """
        mu, std = 0., None
        if test_input is not None:
            if self.rbf is None:
                raise Exception('Must provide prior with embedding layer evaluations for nonstationary case.')
            rbf = self.rbf[int(test_input), :].unsqueeze(0)

        # NOTE: parameter tensor names have the form (e.g.) "layers.hidden_X.W" or "output_layer.W", whereas
        #       hyperparameter tensor names have the same form with (e.g.) ".W_rho_coeffs" instead of ".W"

        if '.W' in name:
            if name.replace('.W', '.W_rho_coeffs') in self.params.keys():
                std = F.softplus(
                    torch.tensordot(rbf, self.params[name.replace('.W', '.W_rho_coeffs')], dims=([1], [0]))).squeeze()
            elif name.replace('.W', '.W_rho') in self.params.keys():
                std = F.softplus(self.params[name.replace('.W', '.W_rho')])
            if name.replace('.W', '.W_mu_coeffs') in self.params.keys():
                mu = torch.tensordot(rbf, self.params[name.replace('.W', '.W_mu_coeffs')], dims=([1], [0])).squeeze()
            elif name.replace('.W', '.W_mu') in self.params.keys():
                mu = self.params[name.replace('.W', '.W_mu')]
        elif '.b' in name:
            if name.replace('.b', '.b_rho_coeffs') in self.params.keys():
                std = F.softplus(
                    torch.tensordot(rbf, self.params[name.replace('.b', '.b_rho_coeffs')], dims=([1], [0]))).squeeze()
            elif name.replace('.b', '.b_rho') in self.params.keys():
                std = F.softplus(self.params[name.replace('.b', '.b_rho')])
            if name.replace('.b', '.b_mu_coeffs') in self.params.keys():
                mu = torch.tensordot(rbf, self.params[name.replace('.b', '.b_mu_coeffs')], dims=([1], [0])).squeeze()
            elif name.replace('.b', '.b_mu') in self.params.keys():
                mu = self.params[name.replace('.b', '.b_mu')]

        return mu, std

    def logp(self, net, test_input=None):
        """
        Compute log joint prior.

        :param net: nn.Module, the input network to be evaluated
        :param test_input: int, specifies row index of test input in the (n_test_h*n_test_v, input_dim) array
        :return: torch.Tensor, log joint prior
        """
        res = 0.
        for name, param in net.named_parameters():  # param tensors contain weights and biases
            if 'batch_norm' in name:
                continue
            mu, std = self._get_params_by_name(name, test_input)
            var = std ** 2
            res -= 0.5 * torch.sum(((param - mu) ** 2) / var)
        return res

"""
Hierarchical Gaussian prior (Fixed and GPi-H)
"""

class FixedHierarchicalPrior(PriorModule):
    def __init__(self, net, mu=0.0, shape=1.0, rate=1.0):
        """
        Child class for hierarchical Gaussian prior over parameters, with inv-gamma hyperprior over each variance.

        :param net: nn.Module, the input network on which to apply the prior
        :param mu: float, mean parameter for each conditional Gaussian
        :param shape: float, shape parameter for each inv-gamma
        :param rate: float, rate parameter for each inv-gamma
        """
        super().__init__()

        self.hyperprior = True
        self.mu = mu
        self.shape = torch.Tensor([shape])
        self.rate = torch.Tensor([rate])

        self.params = {}
        self._initialise(net)  # initialise the std deviations (sampled from hyperprior)

    def _sample_std(self, shape, rate):
        """
        Sample std dev for layer from inv-gamma with specified parameters.

        :param shape: float, shape parameter for inv-gamma
        :param rate: float, rate parameter for inv-gamma
        :return: torch.Tensor, std dev for layer
        """
        with torch.no_grad():
            gamma_dist = dist.Gamma(shape, rate)
            inv_var = gamma_dist.rsample()  # draw reciprocal variance from gamma with same shape and rate
            std = 1. / (torch.sqrt(inv_var) + 1e-10)  # obtain approximate std dev
            return std

    def resample(self, net):
        """
        Resample std dev of Gaussian prior for all layers using a Gibbs sampler (draw from posterior given parameters).

        :param net: nn.Module, input network for which we want to alter the prior over parameters

        Note: the posterior is as given in eq. 31 in [1]

        [1] Tran et al. 2022 (All you need is a good functional prior for Bayesian deep learning)
        """
        for name, param in net.named_parameters():
            if 'batch_norm' in name:
                continue
            if ('.W' in name) or ('.b' in name):
                sumcnt = param.detach().numel()  # add to shape to obtain posterior shape
                sumsqr = (param.detach() ** 2).sum().item()  # add to rate to obtain posterior rate

                shape_ = self.shape + 0.5 * sumcnt  # posterior shape
                rate_ = self.rate + 0.5 * sumsqr  # posterior rate
                std = self._sample_std(shape_, rate_)  # resample std dev for this layer

                if '.W' in name:
                    self.params[name.replace('.W', '.W_std')] = std
                if '.b' in name:
                    self.params[name.replace('.b', '.b_std')] = std

    def _initialise(self, net):
        """
        Initialise network by sampling std dev for each layer.

        :param net: nn.Module, input network to be initialised
        """
        for name, param in net.named_parameters():
            if 'batch_norm' in name:
                continue
            if '.W' in name:
                self.params[name.replace('.W', '.W_std')] = self._sample_std(self.shape, self.rate)
            elif '.b' in name:
                self.params[name.replace('.b', '.b_std')] = self._sample_std(self.shape, self.rate)

    def _get_params_by_name(self, name):
        """
        Extract hyperparameters for layer by specifying name of corresponding parameters.

        :param name: str, name of parameters
        :return: tuple, 2*(float) or 2*(torch.Tensor), mean and std dev for the layer's parameters
        """
        std = None
        if '.W' in name:
            if name.replace('.W', '.W_std') in self.params.keys():
                std = self.params[name.replace('.W', '.W_std')]
        elif '.b' in name:
            if name.replace('.b', '.b_std') in self.params.keys():
                std = self.params[name.replace('.b', '.b_std')]

        return self.mu, std

    def logp(self, net, test_input=None):
        """
        Compute log joint prior.

        :param net: nn.Module, the input network to be evaluated
        :return: torch.Tensor, log joint prior
        """
        res = 0.
        for name, param in net.named_parameters():
            if 'batch_norm' in name:
                continue
            mu, std = self._get_params_by_name(name)
            if std is None:
                continue
            var = std ** 2
            res -= 0.5 * torch.sum((param - mu) ** 2 / var)
        return res

class OptimHierarchicalPrior(PriorModule):
    def __init__(self, saved_path, rbf=None, device="cpu"):
        """
        Child class for optimised hierarchical Gaussian prior over parameters, with inv-gamma hyperprior (GPi-H).

        :param saved_path: str, path to checkpoint containing optimised parameters
        :param rbf: torch.Tensor, embedding layer evaluated on all spatial inputs
        :param device: str, specify device for module
        """
        super().__init__()

        self.hyperprior = True  # option used in bayes_net.py to identify hierarchical prior
        self.params = {}
        self.device = device
        self.rbf = rbf

        data = torch.load(saved_path, map_location=torch.device(self.device))
        for name, param in data.items():
            self.params[name] = param.to(self.device)

    def to(self, device):
        """
        Move each network parameter to the configured device.

        :param device: str, specify device to transfer to
        :return: instance of OptimHierarchicalPrior
        """
        for name in self.params.keys():
            self.params[name] = self.params[name].to(device)
        return self

    def _sample_std(self, shape, rate):
        """
        Sample std dev for layer from inv-gamma with specified parameters.

        :param shape: float, shape parameter for inv-gamma
        :param rate: float, rate parameter for inv-gamma
        :return: torch.Tensor, std dev for layer
        """
        with torch.no_grad():
            gamma_dist = dist.Gamma(shape, rate)
            inv_var = gamma_dist.rsample()
            std = 1. / (torch.sqrt(inv_var) + 1e-10)

            return std

    def resample(self, net, test_input=None):
        """
        Resample std dev of Gaussian prior for all layers using a Gibbs sampler (draw from posterior given parameters).

        :param net: nn.Module, input network for which we want to alter the prior over parameters

        Note: the posterior is as given in eq. 31 in [1]

        [1] Tran et al. 2022 (All you need is a good functional prior for Bayesian deep learning)
        """
        if test_input is not None:
            if self.rbf is None:
                raise Exception('Must provide prior with embedding layer evaluations for nonstationary case.')
            rbf = self.rbf[int(test_input), :].reshape(1, -1)

        for name, param in net.named_parameters():
            if 'batch_norm' in name:
                continue
            if ('.W' in name) or ('.b' in name):
                sumcnt = param.detach().numel()
                sumsqr = (param.detach() ** 2).sum().item()

                if '.W' in name:
                    if name.replace('.W', '.W_shape_coeffs') in self.params.keys():
                        shape = F.softplus(
                            torch.tensordot(rbf, self.params[name.replace('.W', '.W_shape_coeffs')],
                                            dims=([1], [0]))).squeeze()
                    elif name.replace('.W', '.W_shape') in self.params.keys():
                        shape = F.softplus(self.params[name.replace('.W', '.W_shape')])
                    if name.replace('.W', '.W_rate_coeffs') in self.params.keys():
                        rate = F.softplus(
                            torch.tensordot(rbf, self.params[name.replace('.W', '.W_rate_coeffs')],
                                            dims=([1], [0]))).squeeze()
                    elif name.replace('.W', '.W_rate') in self.params.keys():
                        rate = F.softplus(self.params[name.replace('.W', '.W_rate')])
                elif '.b' in name:
                    if name.replace('.b', '.b_shape_coeffs') in self.params.keys():
                        shape = F.softplus(
                            torch.tensordot(rbf, self.params[name.replace('.b', '.b_shape_coeffs')],
                                            dims=([1], [0]))).squeeze()
                    elif name.replace('.b', '.b_shape') in self.params.keys():
                        shape = F.softplus(self.params[name.replace('.b', '.b_shape')])
                    if name.replace('.b', '.b_rate_coeffs') in self.params.keys():
                        rate = F.softplus(
                            torch.tensordot(rbf, self.params[name.replace('.b', '.b_rate_coeffs')],
                                            dims=([1], [0]))).squeeze()
                    elif name.replace('.b', '.b_rate') in self.params.keys():
                        rate = F.softplus(self.params[name.replace('.b', '.b_rate')])

                shape_ = shape + 0.5 * sumcnt
                rate_ = rate + 0.5 * sumsqr
                std = self._sample_std(shape_, rate_)

                if '.W' in name:
                    self.params[name.replace('.W', '.W_std')] = std
                if '.b' in name:
                    self.params[name.replace('.b', '.b_std')] = std

    def _initialise(self, net, test_input=None):
        """
        Initialise network by sampling std dev for each layer.

        :param net: nn.Module, input network to be initialised
        """
        if test_input is not None:
            if self.rbf is None:
                raise Exception('Must provide prior with embedding layer evaluations for nonstationary case.')
            rbf = self.rbf[int(test_input), :].reshape(1, -1)

        for name, param in net.named_parameters():
            if 'batch_norm' in name:
                continue
            if '.W' in name:
                if name.replace('.W', '.W_shape_coeffs') in self.params.keys():
                    shape = F.softplus(
                        torch.tensordot(rbf, self.params[name.replace('.W', '.W_shape_coeffs')],
                                        dims=([1], [0]))).squeeze()
                elif name.replace('.W', '.W_shape') in self.params.keys():
                    shape = F.softplus(self.params[name.replace('.W', '.W_shape')])
                if name.replace('.W', '.W_rate_coeffs') in self.params.keys():
                    rate = F.softplus(
                        torch.tensordot(rbf, self.params[name.replace('.W', '.W_rate_coeffs')],
                                        dims=([1], [0]))).squeeze()
                elif name.replace('.W', '.W_rate') in self.params.keys():
                    rate = F.softplus(self.params[name.replace('.W', '.W_rate')])
                self.params[name.replace('.W', '.W_std')] = self._sample_std(shape, rate)
            elif '.b' in name:
                if name.replace('.b', '.b_shape_coeffs') in self.params.keys():
                    shape = F.softplus(
                        torch.tensordot(rbf, self.params[name.replace('.b', '.b_shape_coeffs')],
                                        dims=([1], [0]))).squeeze()
                elif name.replace('.b', '.b_shape') in self.params.keys():
                    shape = F.softplus(self.params[name.replace('.b', '.b_shape')])
                if name.replace('.b', '.b_rate_coeffs') in self.params.keys():
                    rate = F.softplus(
                        torch.tensordot(rbf, self.params[name.replace('.b', '.b_rate_coeffs')],
                                        dims=([1], [0]))).squeeze()
                elif name.replace('.b', '.b_rate') in self.params.keys():
                    rate = F.softplus(self.params[name.replace('.b', '.b_rate')])
                self.params[name.replace('.b', '.b_std')] = self._sample_std(shape, rate)

    def _get_params_by_name(self, name, test_input=None):
        """
        Extract hyperparameters for layer by specifying name of corresponding parameters.

        :param name: str, name of parameters
        :param test_input: int, specifies row index of test input
        :return: tuple, 2*(float) or 2*(torch.Tensor), mean and std dev for the layer's parameters
        """
        mu, std = 0., None
        if test_input is not None:
            if self.rbf is None:
                raise Exception('Must provide prior with embedding layer evaluations for nonstationary case.')
            rbf = self.rbf[int(test_input), :].reshape(1, -1)

        if '.W' in name:
            if name.replace('.W', '.W_std') in self.params.keys():
                std = self.params[name.replace('.W', '.W_std')]
            if name.replace('.W', '.W_mu_coeffs') in self.params.keys():
                mu = torch.tensordot(rbf, self.params[name.replace('.W', '.W_mu_coeffs')], dims=([1], [0])).squeeze()
            elif name.replace('.W', '.W_mu') in self.params.keys():
                mu = self.params[name.replace('.W', '.W_mu')]
        elif '.b' in name:
            if name.replace('.b', '.b_std') in self.params.keys():
                std = self.params[name.replace('.b', '.b_std')]
            if name.replace('.b', '.b_mu_coeffs') in self.params.keys():
                mu = torch.tensordot(rbf, self.params[name.replace('.b', '.b_mu_coeffs')], dims=([1], [0])).squeeze()
            elif name.replace('.b', '.b_mu') in self.params.keys():
                mu = self.params[name.replace('.b', '.b_mu')]

        return mu, std

    def logp(self, net, test_input=None):
        """
        Compute log joint prior.

        :param net: nn.Module, the input network to be evaluated
        :return: torch.Tensor, log joint prior
        """
        res = 0.
        for name, param in net.named_parameters():
            if 'batch_norm' in name:
                continue
            mu, std = self._get_params_by_name(name, test_input)
            if std is None:
                continue
            var = std ** 2
            res -= 0.5 * torch.sum((param - mu) ** 2 / var)
        return res

