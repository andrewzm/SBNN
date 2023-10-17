"""
Functions for analysing time series data
"""

import numpy as np


def acv(x, lag):
    """
    Compute sample autocovariance for time series.

    :param x: np.ndarray, time series vector
    :param lag: int, signed lag value
    :return: float, autocovariance function value gamma_h
    """
    h = abs(lag)
    n = len(x)
    xbar = np.mean(x)

    autocov = 0
    for t in range(n - h):
        autocov += (x[t+h] - xbar) * (x[t] - xbar)
    autocov /= n

    return autocov

def acf(x, lag, return_all=False, step=1):
    """
    Compute sample autocorrelation for time series.

    :param x: np.ndarray, time series vector
    :param lag: int, signed lag value
    :param return_all: bool, specifies whether all lower rho_h values should be returned
    :param step: int, skip (step-1) lags when computing acf, default step=1 (compute for each lag)
    :return: float, autocorrelation function value rho_h
    """
    if not return_all:
        return acv(x, lag) / acv(x, 0)

    h_range = np.arange(0, lag+1, step)

    return [acv(x, h) / acv(x, 0) for h in h_range]

def pacf(x, lag, return_all=False):
    """
    Compute sample partial autocorrelation for time series using Durbin-Levinson Algorithm.

    :param x: np.ndarray, time series vector
    :param lag: int, signed lag value
    :param return_all: bool, specifies whether all lower phi_kk values should be returned
    :return: float, partial autocorrelation function value(s) phi_kk
    """
    k = abs(lag)
    if k <= 1:
        return acf(x, lag)

    phi = np.zeros((k, k))  # row for each Durbin-Levinson step; phi_ij in (i,j) entry / [i-1,j-1] index
    phi[0, 0] = acf(x, 1)  # phi_11 value
    for n in range(2, k+1):
        ni = n-1  # n index

        # Numerator value of phi_nn
        phi_num = acf(x, n)
        for j in range(1, n):
            ji = j-1  # j index
            phi_num -= phi[ni-1, ji] * acf(x, n-j)

        # Denominator value of phi_nn
        phi_den = 1
        for j in range(1, n):
            ji = j-1  # j index
            phi_den -= phi[ni-1, ji] * acf(x, j)

        # Store phi_nn value
        phi[ni, ni] = phi_num / phi_den

        # Compute phi_nj values for j=1,2,...,n-1
        for j in range(1, n):
            ji = j-1  # j index
            phi[ni, ji] = phi[ni-1, ji] - phi[ni, ni] * phi[ni-1, n-j-1]  # careful with second index in final term

    # Return phi_kk, optionally with earlier phi_nn values
    if not return_all:
        return phi[k-1, k-1]

    return [1] + list(np.diag(phi))

def ess(x, lag):
    """
    Compute effective sample size for time series using ACF values up to specified lag.

    :param x: np.ndarray, time series vector
    :param lag: int, maximum lag to compute ACFs (higher maximum gives more accurate ESS value, although the maximal
        lag reasonable is a function of the length of the time series vector)
    :return: float, effective sample size of the time series (not rounded)
    """
    acf_sum = 0
    for k in range(lag):
        acf_sum += acf(x, lag=k+1, return_all=False)
    n = len(x)
    return min([n, n / (1 + 2 * acf_sum)])

def max_corr_lag(x):
    """
    Compute maximum sample ACF lag for given time series, such that correlations are not yet excessively noisy.

    :param x: np.ndarray, time series vector
    :return: int, maximum sample correlation
    """
    n = len(x)
    half_n = int(np.floor(n/2))

    # Find first odd lag T such that subsequent two correlations have a negative sum; use T as max lag
    rho_values = acf(x, lag=n, return_all=True)
    for t in range(1, half_n):
        T = int(2*t - 1)
        rho1 = rho_values[T+1]
        rho2 = rho_values[T+2]
        if rho1 + rho2 < 0:
            return T

    # If no such T exists, use half the time series length as maximum lag
    return half_n
