"""
Gaussian process kernel base classes
"""

import torch
import numpy as np
import torch.linalg as la
import scipy.spatial as sps

# Ran into problems using scipy's distance_matrix on GPU (25/08)

class Zero(torch.nn.Module):
    def forward(self, X):
        """
        Compute zero mean function at specified inputs.

        :param X: torch.Tensor, input points
        :return: torch.Tensor, corresponding zero-valued outputs
        """
        return torch.zeros(X.size(0), 1, dtype=X.dtype, device=X.device)

class Kernel(torch.nn.Module):
    def __init__(self, variance, lengthscale):
        """
        Parent class for kernel (covariance) functions.

        :param variance: float, variance parameter (affects amplitude)
        :param lengthscale: float, lengthscale parameter (affects average number of upcrossings of level zero)
        """
        super().__init__()
        self.leng = torch.tensor([lengthscale], dtype=torch.double)
        self.ampl = torch.tensor([variance], dtype=torch.double)

    def square_dist(self, X, X2=None):
        """
        Compute squared distance matrix.

        :param X: torch.Tensor, first set of input points
        :param X2: torch.Tensor, second set of input points
        :return: torch.Tensor, squared distances between X and X2
        """
        if X2 is None:
            X2 = X.clone()

        # Increase precision of calculations
        X = X.to(dtype=torch.float64)
        X2 = X2.to(dtype=torch.float64)

        # Compute squared Euclidean norm at each point
        Xs = (X**2).sum(1)
        X2s = (X2**2).sum(1)

        # Compute distance matrix based on expanded expression
        dist = -2 * torch.matmul(X, X2.t())
        dist += Xs.view(-1, 1) + X2s.view(1, -1)
        return torch.clamp(dist, min=0)  # avoid negative values, due to numerical error

    def euclid_dist(self, X, X2=None, manual=True):
        """
        Compute Euclidean distance matrix (sqrt of above).

        :param X: torch.Tensor, first set of input points
        :param X2: torch.Tensor, second set of input points
        :param manual: bool, specify if distance matrix is computed using manually built function
        :return: torch.Tensor, Euclidean distances between X and X2
        """
        if manual:
            return torch.sqrt(self.square_dist(X, X2))
        device = X.device
        if X2 is None:
            X2 = X
        return torch.from_numpy(sps.distance_matrix(X.cpu(), X2.cpu())).to(device)

    def disp_mx(self, X, X2=None):
        """
        Compute the displacement matrix (note: direction is important).

        :param X: torch.Tensor, first set of input points
        :param X2: torch.Tensor, second set of input points
        :return: torch.Tensor, displacements between points in X
        """
        if X2 is None:
            X2 = X.clone()

        disp1 = X[:, 0].reshape(-1, 1) - X2[:, 0]  # displacements for first coordinate
        disp2 = X[:, 1].reshape(-1, 1) - X2[:, 1]  # displacements for second coordinate
        disp = torch.stack([disp1, disp2], dim=2)  # last index [:, :, i] for each displacement matrix
        disp = torch.unsqueeze(disp, dim=3)  # add extra dimension onto the end

        # Output displacement array with shape (|X|, |X2|, 2, 1)
        return disp

class Isotropic(Kernel):
    def __init__(self, cov, **params):
        """
        Parent class for isotropic kernels.

        :param cov: isotropic kernel function
        :param params: dict, key-value pairs with values for ampl, leng, [power]
        """
        super().__init__(params['ampl'], params['leng'])
        self.cov = cov
        self.params = params

    def K(self, X, X2=None):
        """
        Compute covariance matrix using isotropic kernel.

        :param X: torch.Tensor, first set of input points
        :param X2: torch.Tensor, second set of input points
        :return: torch.Tensor, covariance matrix for X
        """
        r = self.euclid_dist(X, X2)
        return self.cov(r, **self.params)

    def K2(self, X, X2=None):
        return self.euclid_dist(X, X2)

class Nonstationary(Kernel):
    def __init__(self, cov, x0, **params):
        """
        Parent class for nonstationary kernels (using generalised isotropic form).

        :param cov, isotropic kernel function to use for nonstationary kernel construction
        :param x0: tuple, point around which to construct nonstationarity
        :param params: dict, key-value pairs with values for ampl, leng, [power]
        """
        super().__init__(params['ampl'], params['leng'])
        self.cov = cov
        self.x0 = x0
        self.params = params

    def K(self, X, X2=None):
        """
        Compute covariance matrix using nonstationary kernel.

        :param X: torch.Tensor, first set of input points
        :param X2: torch.Tensor, second set of input points
        :return: torch.Tensor, covariance matrix for X
        """
        # Compute displacement matrix, and distances from x0
        if X2 is None:
            X2 = X.clone()
        disp = self.disp_mx(X, X2).float()
        x0_dist1 = torch.sqrt((X[:, 0] - self.x0[0]) ** 2 + (X[:, 1] - self.x0[1]) ** 2).cpu().numpy()
        x0_dist2 = torch.sqrt((X2[:, 0] - self.x0[0]) ** 2 + (X2[:, 1] - self.x0[1]) ** 2).cpu().numpy()

        # Compute number of spatial points in each input set
        n1 = X.shape[0]
        n2 = X2.shape[0]

        # Compute Sigma_i matrix, shape (2, 2), for each spatial point
        s11 = s22 = np.exp(x0_dist1)
        s21 = s12 = np.zeros(n1)
        s11_2 = s22_2 = np.exp(x0_dist2)
        s21_2 = s12_2 = np.zeros(n2)

        # Arrange Sigma_i matrices into arrays, shapes (n1, 2, 2) and (n2, 2, 2)
        Sigma_all = torch.permute(torch.Tensor([[s11, s12], [s21, s22]]), dims=(2, 0, 1)).to(disp.device)
        Sigma2_all = torch.permute(torch.Tensor([[s11_2, s12_2], [s21_2, s22_2]]), dims=(2, 0, 1)).to(disp.device)

        # Restrict values to be non-negative, to handle numerical error
        Sigma_all = torch.clamp(Sigma_all, min=0)
        Sigma2_all = torch.clamp(Sigma2_all, min=0)

        # Compute arrays needed for each pairwise covariance calculations, shape (n1, n2, 2, 2)
        Sigma_i = torch.tile(torch.unsqueeze(Sigma_all, dim=1), dims=(1, n2, 1, 1))
        Sigma_j = torch.tile(torch.unsqueeze(Sigma2_all, dim=0), dims=(n1, 1, 1, 1))
        Sigma_i_plus_j_on_2 = torch.clamp((Sigma_i + Sigma_j) / 2, min=0)

        # The below is an efficient way of constructing Q in (6)
        disp_T = torch.permute(disp, dims=(0, 1, 3, 2))  # transpose last two dimensions
        if disp.shape[1] != Sigma_i_plus_j_on_2.shape[1] or disp.shape[0] != Sigma_i_plus_j_on_2.shape[0]:
            print(disp.shape)
            print(Sigma_i_plus_j_on_2.shape)
        Q = torch.squeeze(disp_T @ la.inv(Sigma_i_plus_j_on_2) @ disp)

        # Compute covariance matrix for X
        logdet = 0.25 * torch.logdet(Sigma_i) + 0.25 * torch.logdet(Sigma_j) - 0.5 * torch.logdet(Sigma_i_plus_j_on_2)
        return torch.exp(logdet) * self.cov(torch.sqrt(Q), **self.params)

