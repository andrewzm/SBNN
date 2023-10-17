"""
Measurement set generators
"""

import torch
import numpy as np


class GridGenerator(object):
    def __init__(self, x_min, x_max, input_dim):
        """
        Data generation object, generate grid of values for measurement set.

        :param x_min: float, minimal input value
        :param x_max: float, maximal input value
        :param input_dim: int, specify input dimensions
        """
        self.x_min = x_min
        self.x_max = x_max
        self.input_dim = input_dim

    def get(self, n_data):
        """
        Obtain measurement set between x_min and x_max containing ~n_data points.

        Note: In 1D, the set contains exactly n_data points. In 2D, the measurement set contains approx. n_data points,
        unless n_data is a perfect square, in which case the set again has exactly n_data points.

        :param n_data: int, size of measurement set
        :return: torch.Tensor, size (~n_data, input_dim), measurement set
        """
        if self.input_dim == 1:
            X = torch.linspace(self.x_min, self.x_max, n_data)
            return X.reshape([-1, 1])
        elif self.input_dim == 2:
            X = np.linspace(self.x_min, self.x_max, round(np.sqrt(n_data)))
            Xs, Ys = np.meshgrid(X, X, indexing='xy')  # Cartesian indexing (xy) is default
            XY = np.vstack((Xs.flatten(), Ys.flatten())).T  # col for X and col for Y
            return torch.from_numpy(XY).float()
        else:
            raise Exception('Only 1D and 2D input dimensions are implemented.')
