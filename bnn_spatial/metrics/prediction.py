"""
Prediction performance metrics
"""

import numpy as np
import torch
from scipy.special import ndtri

# Note: ndtri stands for normal (n) distribution (dtr) inverse (i)

# bnn_preds = [n_samples, n_test]
# y = [n_holdout]
# bnn_preds[:, inds] = [n_samples, n_holdout]

def rmspe(preds, obs, return_all=False):
    """
    Compute root mean-squared prediction error.

    :param preds: np.ndarray, shape (n_samples, n_holdout), holdout set predictions
    :param obs: np.ndarray, shape (n_holdout), holdout set observational targets
    :param return_all: bool, specify if RMSPE is given for each MCMC sample (otherwise average is given)
    :return: np.ndarray or float, holdout set RMSPE
    """
    sq_diffs = (preds - np.expand_dims(obs, 0)) ** 2
    sample_rmspe = np.sqrt(np.average(sq_diffs, axis=1))
    if return_all:
        return sample_rmspe
    else:
        return np.average(sample_rmspe)

def empirical_quantile(preds, me_var, alpha=(0.05, 0.95)):
    """
    Obtain quantile(s) from empirical distribution at each spatial point.

    :param preds: np.ndarray, shape (n_samples, n_holdout), holdout set predictions
    :param me_var: float, measurement error variance
    :param alpha: tuple (float), probabilities for quantiles to cut off below
    :return: np.ndarray, empirical quantile(s) for each holdout set location, cutting off prob in the tail(s)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    n_samples = preds.shape[0]
    n_holdout = preds.shape[1]
    noise = np.random.randn(n_samples, n_holdout) * np.sqrt(me_var)  # introduce white noise
    noisy_preds = preds + noise
    pred_quantiles = np.quantile(noisy_preds, q=alpha, axis=0)
    return pred_quantiles

def perc_coverage(preds, obs, me_var, percent=90):
    """
    Compute X-percent coverage (default X = 90).

    :param preds: np.ndarray, shape (n_samples, n_holdout), holdout set predictions
    :param obs: np.ndarray, shape (n_holdout), holdout set observational targets
    :param me_var: float, measurement error variance
    :param percent: float, specify X for X-percent coverage (default 90)
    :return: np.ndarray or float, holdout set X-percent coverage
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()

    tail_prob = 1 - percent / 100
    alpha_levels = (tail_prob/2, 1-tail_prob/2)
    pred_quantiles = empirical_quantile(preds=preds,
                                        me_var=me_var,
                                        alpha=alpha_levels)
    l = pred_quantiles[0, :].squeeze()  # lower bound of prediction interval
    u = pred_quantiles[1, :].squeeze()  # upper bound of prediction interval
    y = obs.squeeze()
    indicator = ((y >= l) & (y <= u))
    return np.average(indicator)

def interval_score(preds, obs, me_var, alpha=0.1):
    """
    Compute negatively-oriented interval score.

    :param preds: np.ndarray, shape (n_samples, n_holdout), holdout set predictions
    :param obs: np.ndarray, shape (n_holdout), holdout set observational targets
    :param me_var: float, measurement error variance
    :param alpha: float, probability of tail region
    :return: np.ndarray or float, holdout set interval score
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()

    alpha_levels = (alpha/2, 1-alpha/2)
    pred_quantiles = empirical_quantile(preds=preds,
                                        me_var=me_var,
                                        alpha=alpha_levels)
    l = pred_quantiles[0, :].squeeze()  # lower bound of prediction interval
    u = pred_quantiles[1, :].squeeze()  # upper bound of prediction interval
    y = obs.squeeze()
    score = (u - l) + (2/alpha) * ((l - y) * (y < l) + (y - u) * (y > u))
    return np.average(score)


