"""
Utility functions for normalising and unnormalising data
"""

import numpy as np


def zscore_normalisation(X, mean=None, std=None, eps=1e-10):
    """
    Apply Z-score normalisation on given data.

    :param X: np.ndarray, shape (batch_size, num_dims), input data
    :param mean: np.ndarray, shape (num_dims), the given mean of the data
    :param std: np.ndarray, shape (num_dims), the given std dev of the data
    :param eps: float, for numerical stability
    :return: tuple, normalised input data, mean, std dev
    """
    if X is None:
        return None, None, None

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalised = (X - mean) / (std + eps)

    return X_normalised, mean, std

def zscore_unnormalisation(X_normalised, mean, std):
    """
    Apply Z-score unnormalisation to given normalised data.

    :param X_normalised: np.ndarray, shape (batch_size, num_dims), input data to unnormalise
    :param mean: np.ndarray, shape (num_dims), the given mean of the data
    :param std: np.ndarray, shape (num_dims), the given std dev of the data
    :return: np.ndarray, shape (batch_size, num_dims), unnormalised input data
    """
    X_unnormalised = X_normalised * std + mean
    return X_unnormalised
