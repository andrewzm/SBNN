"""
Plotting functions
"""

import torch
import numpy as np
import random as rand
import math

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtrans
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.special import ndtr
from itertools import product
import seaborn as sb
import pandas as pd

from ..metrics.sampling import compute_rhat
from ..bnn.activation_fns import rbf_scale
from ..metrics.timeseries import acf, pacf, ess, max_corr_lag

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # to suppress FutureWarning caused by mtrans.Bbox

plt.rcParams['figure.figsize'] = [14, 7]
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['image.aspect'] = 'auto'
#plt.rcParams['text.usetex'] = True

#mpl.use('TkAgg')
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def plot_samples(domain, samples, n_keep=10, color="xkcd:bluish", quantile=False):
    """
    For plotting samples from 1D BNN/GP.

    :param domain: torch.Tensor, inputs for sampling
    :param samples: np.ndarray, contains sampled functions
    :param n_keep: int, number of samples to keep in plot
    :param color: str, specify colour for samples and shading
    :param quantile: bool, specify whether we use quantiles for uncertainty bands
    """
    domain = domain.detach().cpu()
    n_samples = samples.shape[1]  # n_plot for BNN/GP; number of outer loops for Lipschitz function
    keep_idx = np.random.permutation(n_samples)[:n_keep]  # randomly choose n_keep sampled functions
    mu = samples.mean(1)  # mean of all sampled functions (mean for each row; functions stored by column)

    if not quantile:
        moe = 2 * samples.std(1)  # 2 std dev margin of error
        ub, lb = mu + moe, mu - moe  # uncertainty bands of +/- 2 std dev
    else:
        q = 100 * ndtr(2)  # obtain % chance given by +2 std dev in normal cdf
        Q = np.percentile(samples, [100 - q, q], axis=1)  # compute (100-q)-th and q-th sample percentiles
        ub, lb = Q[1, :], Q[0, :]  # uncertainty bands of +/- 2 std dev

    plt.figure()
    ax = plt.gca()
    ax.fill_between(domain.flatten(), ub, lb, color=color, alpha=0.25, lw=0)  # shaded region
    ax.plot(domain, (ub+lb)/2, color='xkcd:red', ls=':')  # average of both 2SD bounds (should overlap with mean)
    ax.plot(domain, samples[:, keep_idx], lw=0.75, color=color, alpha=0.8)  # plot randomly chosen samples
    ax.plot(domain, mu, color='xkcd:red')  # mean of all samples

def plot_percentiles(domain, samples, n_keep=10, color="xkcd:bluish", n_percentiles=10, method='median_unbiased'):
    """
    For plotting samples from 1D BNN/GP, overlaying percentile bounds to better illustrate variation.

    :param domain: torch.Tensor, inputs for sampling
    :param samples: np.ndarray, contains sampled functions
    :param n_keep: int, number of samples to keep in plot
    :param color: str, specify colour for samples and shading
    :param n_percentiles: int, number of percentile bounds to overlay
    :param method: str, specify method for computing percentiles
    """
    domain = domain.detach().cpu()
    n_samples = samples.shape[1]  # n_plot for BNN/GP; number of outer loops for Lipschitz function
    keep_idx = np.random.permutation(n_samples)[:n_keep]  # randomly choose n_keep sampled functions
    mu = samples.mean(1)  # mean of all sampled functions (mean for each row; functions stored by column)

    bounds = np.zeros((samples.shape[0], n_percentiles, 2))
    q_range = np.arange(50/n_percentiles, 50, 50/n_percentiles)
    for q_idx, q in enumerate(q_range):
        Q = np.percentile(samples, [100 - q, q], axis=1, method=method)
        ub, lb = Q[1, :], Q[0, :]
        bounds[:, q_idx, 0] = ub
        bounds[:, q_idx, 1] = lb

    plt.figure()
    ax = plt.gca()
    for qq in range(n_percentiles):
        ub = bounds[:, qq, 0]
        lb = bounds[:, qq, 1]
        ax.fill_between(domain.flatten(), ub, lb, color=color, alpha=0.1, lw=0)  # shaded region
    ax.plot(domain, samples[:, keep_idx], lw=0.75, color=color, alpha=0.8)  # plot randomly chosen samples
    ax.plot(domain, mu, color='xkcd:red')  # mean of all samples

def plot_spread(domain, samples, n_keep=10, color="xkcd:bluish", plot_spread=True, figsize = (6,4)):
    """
    For plotting samples from 1D BNN/GP, overlaying std dev bounds to better illustrate variation.

    :param domain: torch.Tensor, inputs for sampling
    :param samples: np.ndarray, contains sampled functions
    :param n_keep: int, number of samples to keep in plot
    :param color: str, specify colour for samples and shading
    """
    domain = domain.detach().cpu()
    n_samples = samples.shape[1]  # n_plot for BNN/GP; number of outer loops for Lipschitz function
    keep_idx = np.random.permutation(n_samples)[:n_keep]  # randomly choose n_keep sampled functions
    mu = samples.mean(1)  # mean of all sampled functions (mean for each row; functions stored by column)

    bounds = np.zeros((samples.shape[0], 12, 2))
    for sd in range(12):
        moe = (sd + 1) / 4 * samples.std(1)
        ub, lb = mu + moe, mu - moe
        bounds[:, sd, 0] = ub
        bounds[:, sd, 1] = lb

    light_color = "xkcd:light{}".format(color[5:])
    dark_color = "xkcd:dark{}".format(color[5:])
    lw = 0.3

    plt.figure(figsize = figsize)
    ax = plt.gca()
        
    if plot_spread == True:
        for sd in range(12):
            ub = bounds[:, sd, 0]
            lb = bounds[:, sd, 1]

            # Plot shaded region
            ax.fill_between(domain.flatten(), ub, lb, color=color, alpha=0.08, lw=0)

    # Plot random samples (first try light shade; if not available, use standard shade)
    ax.plot(domain, samples[:, keep_idx], lw=lw, color=color)
    ax.plot(domain, mu, lw=lw*3, color='xkcd:red')  # mean of all samples

def plot_lipschitz(inner_steps, outer_steps, samples, type='penalty'):
    """
    For plotting norm of Lipschitz function gradient. Also supports plotting concatenated parameter gradient norms, and
    the Lipschitz losses which are maximised to estimate the Wasserstein distance.

    :param inner_steps: int, number of inner loop optimisation steps
    :param outer_steps: int, number of outer loop optimisation steps
    :param samples: np.ndarray, contains sampled norms/losses (rows for inner steps, cols for outer steps)
    :param type: str, specify "penalty" to plot penalty term gradient norms, "parameter" to plot concatenated
        parameter gradient norms, or "loss" to plot Lipschitz losses
    """
    inner_range = np.arange(1, inner_steps+1)  # range of inner steps

    # Adjust settings for each case as appropriate
    if type == 'penalty':
        alpha = 1
        ylab = 'Norm of Lipschitz Function Gradient'
    elif type == 'parameter':
        alpha = 1
        ylab = 'Norm of Combined Parameter Gradients'
    elif type == 'loss':
        alpha = 1
        ylab = 'Regularised Lipschitz Function Loss'
    else:
        raise Exception('Incorrect type specified: need penalty, parameter, or loss')

    # Plot the gradient norms / losses
    plt.figure()
    ax = plt.gca()
    col = plt.get_cmap(name='Spectral_r')(np.linspace(0, 1, 100))  # create array of colourmap values
    indices_to_plot = np.round(
                        np.linspace(math.floor((outer_steps-1)/2), 
                                                outer_steps-2, num = 100)
                     ).astype(int)
    for ss in range(100):
        plt.plot(inner_range, 
                 samples[:, indices_to_plot[ss]], 
                 lw=0.2, color=col[ss], alpha=alpha)  # colour by loop
    plt.ylabel(ylab)
    plt.xlabel('Iteration (Inner Loop)')

    # Add horizontal line marking unit norm for penalty plot
    if type == 'penalty':
        plt.axhline(y=1, ls='--', c='k')  # mark the unit norm
    else:
        plt.axhline(y=0, ls='--', c='k')  # mark zero

    # Add colourbar to plot
    cmap = mpl.colors.ListedColormap(col)
    norm = mpl.colors.Normalize(vmin=math.floor((outer_steps-1)/2), vmax=outer_steps)  # cmap ranges with outer loops
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # maps scalar data to RGBA
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    cbar = plt.colorbar(mappable=sm, cax=cax)
    cbar.set_label('Iteration (Outer Loop)', rotation=270, labelpad=24)

def plot_rbf_filled(domain, net_width, lenscale):
    """
    Plot the RBFs used in the embedding layer, using filled contour plots in 2D.

    :param domain: np.ndarray, contains all test points (coordinates in columns)
    :param net_width: int, embedding layer width (perfect square for 2D case), equal to number of RBFs
    :param lenscale: float, lengthscale parameter value
    """
    # Obtain limits and coordinates of spatial domain
    input_dim = domain.shape[1]
    n_test = np.size(domain)

    if input_dim == 2:
        x1_min, x1_max = domain[:, 0].min(), domain[:, 0].max()
        x2_min, x2_max = domain[:, 1].min(), domain[:, 1].max()
        n_test_h = np.size(np.unique(domain[:, 0]))
        n_test_v = np.size(np.unique(domain[:, 1]))
        X1_subset = np.linspace(x1_min, x1_max, int(np.sqrt(net_width)))
        X2_subset = np.linspace(x2_min, x2_max, int(np.sqrt(net_width)))
        X1_coords, X2_coords = np.meshgrid(X1_subset, X2_subset)
        test_subset = np.vstack((X1_coords.flatten(), X2_coords.flatten())).T
    elif input_dim == 1:
        x_min, x_max = domain.min(), domain.max()
        test_subset = np.linspace(x_min, x_max, net_width)
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    # Populate array of RBF values
    test_subset = torch.from_numpy(test_subset)
    domain = torch.from_numpy(domain)
    rbf_output = torch.zeros(n_test // input_dim, net_width)  # create array for RBFs (rows inputs, cols RBFs)
    for i, x0 in enumerate(test_subset):  # iterate through row-by-row
        rbf_output[:, i] = rbf_scale(torch.subtract(domain, x0), l=lenscale).squeeze()

    # Note: RBF(|X - X0|) = RBF(X - X0); without norm in argument, got unsuitable shape (2-dim instead of 1-dim)
    # Note 2: domain (*, 2) array cycles through x1 values first, then x2 values
    # Note 3: ndarray.reshape uses C-style by default, which fills row-wise (all row 1, then all row 2, etc.), whereas
    #         F-style fills column-wise (all column 1, then all column 2, etc.)

    # Resize array of RBF values (for 2D case)
    if input_dim == 2:
        rbf_output_ = torch.zeros((n_test_v, n_test_h, net_width))
        for ff in range(net_width):
            rbf = rbf_output[:, ff]
            rbf_ = rbf.reshape(n_test_v, -1)
            rbf_output_[:, :, ff] = rbf_

    # Alter colours, making white transparent
    cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    rand.shuffle(cmaps)  # shuffle sequential colourmaps in place
    n_levels = 10  # desired number of levels in contour plot
    blind_levels = 4
    cols_cl = []
    alpha = 0.2  # set transparency for contour plots and colourbar
    for _, col in enumerate(cmaps):
        new_col = plt.get_cmap(name=col)(np.linspace(0, 1, n_levels))
        for ll in range(n_levels):
            if ll <= blind_levels:
                new_col[ll][-1] = 0  # make lower shades (near white) fully transparent
            else:
                new_col[ll][-1] = alpha  # make other shades semi-transparent
        cols_cl.append(new_col)

    # Create new colourmaps using altered colours
    cmaps_cl = []
    for _, col in enumerate(cols_cl):
        cmaps_cl.append(mpl.colors.ListedColormap(col))

    # Obtain colourmap for the colourbar (using greyscale to illustrate transparency)
    grey_col = plt.get_cmap(name='Greys')(np.linspace(0, 1, n_levels))
    for ll in range(n_levels):
        if ll <= blind_levels:
            grey_col[ll][-1] = 0  # make lower shades (near white) fully transparent
        else:
            grey_col[ll][-1] = np.min([2*alpha, 1])  # make other shades semi-transparent
    grey_cmap = mpl.colors.ListedColormap(grey_col)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)  # RBFs range between 0 and 1
    sm = plt.cm.ScalarMappable(cmap=grey_cmap, norm=norm)  # maps scalar data to RGBA

    # Generate overlaid contour plots of RBFs (cycling through colourmaps)
    plt.figure()
    if input_dim == 2:
        X1, X2 = np.meshgrid(domain[:, 0].unique().numpy(), domain[:, 1].unique().numpy())
        for ff in range(net_width):  # loop for each RBF
            plt.contourf(X1, X2, rbf_output_[:, :, ff], levels=n_levels, cmap=cmaps_cl[ff % len(cmaps)], origin='lower')

            # Add black dot marking centroid for each RBF (which is the maximum)
            ctr_idx = np.unravel_index(np.argmax(rbf_output_[:, :, ff]), shape=(n_test_v, n_test_h))
            ctr_x1 = X1[ctr_idx]  # note if x,y or i,j indexing is used
            ctr_x2 = X2[ctr_idx]
            plt.plot(ctr_x1, ctr_x2, color='k', marker='o', ms=3)
        plt.axis('scaled')
        plt.colorbar(mappable=sm, pad=0.01)
    else:
        for ff in range(net_width):
            plt.plot(domain, rbf_output[:, ff])

def plot_rbf(domain, net_width, lenscale, levels=[0.78]):
    """
    Plot the RBFs used in the embedding layer, using standard contour plots in 2D.

    :param domain: np.ndarray, contains all test points (coordinates in columns)
    :param net_width: int, embedding layer width (perfect square for 2D case), equal to number of RBFs
    :param lenscale: float, lengthscale parameter value
    :param levels: list [float], levels of the RBF to plot contour of (in ascending order)
    """
    # Obtain limits and coordinates of spatial domain
    input_dim = domain.shape[1]
    n_test = np.size(domain)

    if input_dim == 2:
        x1_min, x1_max = domain[:, 0].min(), domain[:, 0].max()
        x2_min, x2_max = domain[:, 1].min(), domain[:, 1].max()
        n_test_h = np.size(np.unique(domain[:, 0]))
        n_test_v = np.size(np.unique(domain[:, 1]))
        X1_subset = np.linspace(x1_min, x1_max, int(np.sqrt(net_width)))
        X2_subset = np.linspace(x2_min, x2_max, int(np.sqrt(net_width)))
        X1_coords, X2_coords = np.meshgrid(X1_subset, X2_subset)
        test_subset = np.vstack((X1_coords.flatten(), X2_coords.flatten())).T
    elif input_dim == 1:
        x_min, x_max = domain.min(), domain.max()
        test_subset = np.linspace(x_min, x_max, net_width)
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    # Populate array of RBF values
    test_subset = torch.from_numpy(test_subset)
    domain = torch.from_numpy(domain)
    rbf_output = torch.zeros(n_test // input_dim, net_width)  # create array for RBFs (rows inputs, cols RBFs)
    for i, x0 in enumerate(test_subset):  # iterate through row-by-row
        rbf_output[:, i] = rbf_scale(torch.subtract(domain, x0), l=lenscale).squeeze()

    # Note: RBF(|X - X0|) = RBF(X - X0); without norm in argument, got unsuitable shape (2-dim instead of 1-dim)
    # Note 2: domain (*, 2) array cycles through x1 values first, then x2 values
    # Note 3: ndarray.reshape uses C-style by default, which fills row-wise (all row 1, then all row 2, etc.), whereas
    #         F-style fills column-wise (all column 1, then all column 2, etc.)

    # Resize array of RBF values (for 2D case)
    if input_dim == 2:
        rbf_output_ = torch.zeros((n_test_v, n_test_h, net_width))
        for ff in range(net_width):
            rbf = rbf_output[:, ff]
            rbf_ = rbf.reshape(n_test_v, -1)
            rbf_output_[:, :, ff] = rbf_

    # Generate overlaid contour plots of RBFs (cycling through colourmaps)
    plt.figure()
    if input_dim == 2:
        X1, X2 = np.meshgrid(domain[:, 0].unique().numpy(), domain[:, 1].unique().numpy())
        for ff in range(net_width):  # loop for each RBF
            plt.contour(X1, X2, rbf_output_[:, :, ff], levels=levels, origin='lower', colors='k')

            # Add centroid marker for each RBF (which is the maximiser)
            ctr_idx = np.argmax(rbf_output[:, ff])
            ctr_x1, ctr_x2 = domain[ctr_idx, :]
            plt.plot(ctr_x1, ctr_x2, color='k', marker='+', ms=10)
        plt.axis('scaled')
    else:
        for ff in range(net_width):
            plt.plot(domain, rbf_output[:, ff])

def plot_param_traces(param_chains, n_chains, net_depth, n_discarded, n_burn, trace_titles, legend_entries):
    """
    Trace plots of network parameter MCMC iterates.

    :param param_chains: np.ndarray, array of MCMC samples (one col per step, one row per parameter)
    :param n_chains: int, number of chains of sampling
    :param net_depth: int, number of layers in neural network containing optimised parameters
    :param n_discarded: int, number of samples discarded after burn-in adaptation phase
    :param n_burn: int, number of samples discarded during burn-in adaptation phase
    :param trace_titles: list, titles of trace subplots
    :param legend_entries: list, legend contents (labels for the chains)
    """
    chain_len = param_chains.shape[1] // n_chains

    # Generate trace plots for each parameter
    fig, ax = plt.subplots(nrows=2, ncols=net_depth)
    for cc in range(n_chains):
        for rr in range(2):
            for vv in range(net_depth):
                ax[rr, vv].plot(param_chains[2 * vv + rr, chain_len * cc:chain_len * (cc + 1)], lw=0.5)
                if cc == 0:
                    # Extract chains used for prediction (used for Rhat computation)
                    chain = param_chains[2 * vv + rr, :]
                    chain_pred_len = chain_len - n_burn - n_discarded
                    chain_pred = np.zeros(chain_pred_len * n_chains)  # MCMC samples used for predictions
                    for ch in range(n_chains):
                        chain_pred[ch * chain_pred_len:(ch + 1) * chain_pred_len] \
                            = chain[ch * chain_len + n_burn + n_discarded:(ch + 1) * chain_len]

                    # Compute Rhat on the samples used for prediction, and add to panel along with other info
                    ax[rr, vv].axvline(x=n_discarded + n_burn, lw=1, ls='--', c='k', label='_nolegend_')
                    param_str = trace_titles[2 * vv + rr]
                    rhat = compute_rhat(chain_pred.reshape(-1, 1), n_chains=n_chains).item()
                    rhat_str = r'$\widehat{{R}}$ = {:.4f}'.format(rhat)
                    ax[rr, vv].set_title(param_str + '\n' + rhat_str)
                    ax[rr, vv].autoscale(enable=True, axis='x', tight=True)
                    #ax[rr, vv].locator_params(axis='x', nbins=3)  # set number of x ticks
                    ax[rr, vv].xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=3))
                    ax[rr, vv].yaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=3))

        # Get the bounding boxes of the axes including text decorations
        fig.tight_layout()
        r = fig.canvas.get_renderer()
        get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

        # Get the minimum and maximum extent, get the coordinate half-way between those
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

        # Draw a horizontal lines at those coordinates
        for y in ys:
            line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
            fig.add_artist(line)

    fig.legend(legend_entries, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=n_chains)

def plot_output_traces(domain, preds, n_chains, n_discarded, n_burn, legend_entries):
    """
    Panelled trace plot(s) of network output at specified input(s), overlaid for multiple chains.

    :param domain: torch.Tensor, set of all test inputs
    :param preds: np.ndarray, network predictions at test inputs (rows MCMC steps, cols inputs)
    :param n_chains: int, number of chains of sampling used
    :param n_discarded: int, number of samples discarded after burn-in adaptation phase
    :param n_burn: int, number of samples discarded during burn-in adaptation phase
    :param legend_entries: list, legend contents (labels for the chains)
    """
    domain = domain.detach().cpu()
    input_dim = domain.shape[1]
    if input_dim == 2:
        x1_range = domain[:, 0].unique(sorted=True).numpy()
        x2_range = domain[:, 1].unique(sorted=True).numpy()
        domain = domain.numpy()
        x1_len = len(x1_range)
        x2_len = len(x2_range)
        x1_idxs = np.round(np.linspace(0, x1_len, 5)[1:-1]).astype(int)
        x2_idxs = np.round(np.linspace(0, x2_len, 4)[1:-1]).astype(int)
        inputs = [[x1_range[x1_idx], x2_range[x2_idx]] for x2_idx, x1_idx in product(x2_idxs, x1_idxs)]
    elif input_dim == 1:
        domain = domain.numpy()
        x_len = len(domain)
        x_idxs = np.round(np.linspace(0, x_len, 8)[1:-1]).astype(int)
        inputs = domain[x_idxs]
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    chains_idx = np.zeros(6, dtype=int)
    for x_idx, x in enumerate(inputs):
        chains_idx[x_idx] = int(np.argwhere(np.all(domain == x, axis=1)).squeeze())
    chains = preds[:, chains_idx].squeeze()  # network output traces
    chain_len = chains.shape[0] // n_chains

    # Generate network output trace plots at specified locations
    fig, ax = plt.subplots(nrows=2, ncols=3)
    for rr in range(2):
        for vv in range(3):
            # Extract chains used for prediction
            chain = chains[:, 3*rr + vv].squeeze()
            chain_pred_len = chain_len - n_burn - n_discarded
            chain_pred = np.zeros(chain_pred_len * n_chains)  # MCMC samples used for predictions
            for ch in range(n_chains):
                chain_pred[ch * chain_pred_len:(ch + 1) * chain_pred_len] \
                    = chain[ch * chain_len + n_burn + n_discarded:(ch + 1) * chain_len]

            # Compute ESS on samples used for prediction
            max_lag = max_corr_lag(chain_pred)
            eff_ss = round(ess(chain_pred, lag=max_lag))
            ess_str = 'ESS = {},'.format(eff_ss)

            # Determine x coordinate
            if input_dim == 2:
                x_coord = tuple(np.round(domain[chains_idx[3*rr + vv], :], 2))
            else:
                x_coord = np.round(domain[chains_idx[3*rr + vv]], 2).item()
            x_str = r'$\mathbf{{x}}$ = {}'.format(x_coord)

            # Compute Rhat on samples used for prediction, and add to panel along with other info
            rhat = compute_rhat(chain_pred.reshape(-1, 1), n_chains=n_chains).item()
            rhat_str = r' $\widehat{{R}}$ = {:.4f}'.format(rhat)
            ax[rr, vv].set_title(x_str + '\n' + ess_str + rhat_str)
            ax[rr, vv].autoscale(enable=True, axis='x', tight=True)
            #ax[rr, vv].locator_params(axis='x', nbins=3)  # set number of x ticks
            ax[rr, vv].xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=4))
            ax[rr, vv].yaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=3))
            for ch in range(n_chains):
                ax[rr, vv].plot(chain[chain_len * ch:chain_len * (ch + 1)], lw=0.5)
                if ch == 0:
                    ax[rr, vv].axvline(x=n_burn + n_discarded, lw=1, ls='--', c='k', label='_nolegend_')

    # Get the bounding boxes of the axes including text decorations
    fig.tight_layout()
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    fig.legend(legend_entries, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=n_chains)

def plot_output_chain(domain, preds, n_chains, n_discarded, n_burn):
    """
    Trace plot(s) of network output at specified input(s), plotted contiguously for multiple chains.

    :param domain: torch.Tensor, set of all test inputs
    :param preds: np.ndarray, network predictions at test inputs (rows MCMC steps, cols inputs)
    :param n_chains: int, number of chains of sampling used
    :param n_discarded: int, number of samples discarded after burn-in adaptation phase
    :param n_burn: int, number of samples discarded during burn-in adaptation phase
    """
    domain = domain.detach().cpu()
    input_dim = domain.shape[1]
    if input_dim == 2:
        x1_range = domain[:, 0].unique(sorted=True).numpy()
        x2_range = domain[:, 1].unique(sorted=True).numpy()
        domain = domain.numpy()
        x1_len = len(x1_range)
        x2_len = len(x2_range)
        x1_idxs = np.round(np.linspace(0, x1_len, 5)[1:-1]).astype(int)
        x2_idxs = np.round(np.linspace(0, x2_len, 4)[1:-1]).astype(int)
        inputs = [[x1_range[x1_idx], x2_range[x2_idx]] for x2_idx, x1_idx in product(x2_idxs, x1_idxs)]
    elif input_dim == 1:
        domain = domain.numpy()
        x_len = len(domain)
        x_idxs = np.round(np.linspace(0, x_len, 8)[1:-1]).astype(int)
        inputs = domain[x_idxs]
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    chains_idx = np.zeros(6, dtype=int)
    for x_idx, x in enumerate(inputs):
        chains_idx[x_idx] = int(np.argwhere(np.all(domain == x, axis=1)).squeeze())
    chains = preds[:, chains_idx].squeeze()  # network output traces
    chain_len = chains.shape[0] // n_chains

    # Generate network output trace plots at specified locations
    fig, ax = plt.subplots(nrows=3, ncols=2)
    for rr in range(3):
        for vv in range(2):
            chain = chains[:, 2*rr + vv].squeeze()
            chain_pred_len = chain_len - n_burn - n_discarded
            chain_pred = [0] * chain_pred_len * n_chains  # MCMC samples used for predictions
            for ch in range(n_chains):
                chain_pred[ch * chain_pred_len:(ch + 1) * chain_pred_len] \
                    = chain[ch * chain_len + n_burn + n_discarded:(ch + 1) * chain_len]
            max_lag = max_corr_lag(chain_pred)
            eff_ss = round(ess(chain_pred, lag=max_lag))
            ax[rr, vv].plot(chain, lw=1)
            if input_dim == 2:
                x_coord = tuple(np.round(domain[chains_idx[2*rr + vv], :], 2))
            else:
                x_coord = np.round(domain[chains_idx[2*rr + vv]], 2).item()
            x_str = r'$\mathbf{{x}}$ = {}'.format(x_coord)
            ess_str = 'ESS = {}'.format(eff_ss)
            ax[rr, vv].set_title(x_str + '\n' + ess_str)
            ax[rr, vv].autoscale(enable=True, axis='x', tight=True)
            for ch in range(n_chains):
                ax[rr, vv].axvline(x=ch * chain_len, lw=1, ls='--', c='k', label='_nolegend_')
                ax[rr, vv].axvline(x=ch * chain_len + n_burn + n_discarded, lw=1, ls=':', c='k', label='_nolegend_')

    # Get the bounding boxes of the axes including text decorations
    fig.tight_layout()
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

def plot_output_hist(domain, preds):
    """
    Histogram(s) of network output MCMC iterates at specified input(s).

    :param domain: torch.Tensor, set of all test inputs
    :param preds: np.ndarray, network predictions at test inputs (rows MCMC steps, cols inputs)
    """
    domain = domain.detach().cpu()
    input_dim = domain.shape[1]
    if input_dim == 2:
        x1_range = domain[:, 0].unique(sorted=True).numpy()
        x2_range = domain[:, 1].unique(sorted=True).numpy()
        domain = domain.numpy()
        x1_len = len(x1_range)
        x2_len = len(x2_range)
        x1_idxs = np.round(np.linspace(0, x1_len, 5)[1:-1]).astype(int)
        x2_idxs = np.round(np.linspace(0, x2_len, 4)[1:-1]).astype(int)
        inputs = [[x1_range[x1_idx], x2_range[x2_idx]] for x2_idx, x1_idx in product(x2_idxs, x1_idxs)]
    elif input_dim == 1:
        domain = domain.numpy()
        x_len = len(domain)
        x_idxs = np.round(np.linspace(0, x_len, 8)[1:-1]).astype(int)
        inputs = domain[x_idxs]
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    chains_idx = np.zeros(6, dtype=int)
    for x_idx, x in enumerate(inputs):
        chains_idx[x_idx] = int(np.argwhere(np.all(domain == x, axis=1)).squeeze())
    chains = preds[:, chains_idx].squeeze()  # network output traces

    # Generate network output histograms at specified locations
    fig, ax = plt.subplots(nrows=2, ncols=3)
    for rr in range(2):
        for vv in range(3):
            chain = chains[:, 3*rr + vv].squeeze()
            max_lag = max_corr_lag(chain)
            eff_ss = round(ess(chain, lag=max_lag))
            ax[rr, vv].hist(chain)
            if input_dim == 2:
                x_coord = tuple(np.round(domain[chains_idx[3*rr + vv], :], 2))
            else:
                x_coord = np.round(domain[chains_idx[3*rr + vv]], 2).item()
            x_str = r'$\mathbf{{x}}$ = {}'.format(x_coord)
            ess_str = 'ESS = {}'.format(eff_ss)
            ax[rr, vv].set_title(x_str + '\n' + ess_str)
            ax[rr, vv].xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=3))

    # Get the bounding boxes of the axes including text decorations
    fig.tight_layout()
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

def plot_output_acf(domain, preds, n_samples_kept, n_chains, acf_step=1):
    """
    Panelled ACF plots of network output MCMC iterates.

    :param domain: torch.Tensor, set of all test inputs
    :param preds: np.ndarray, network predictions at test inputs (rows MCMC steps, cols inputs)
    :param n_samples_kept: int, number of samples retained after discarding burn-in
    :param n_chains: int, number of chains of sampling used
    :param acf_step: int, interval at which to plot the ACF (default 1, plotting each ACF value)
    """
    domain = domain.detach().cpu()
    input_dim = domain.shape[1]
    if input_dim == 2:
        x1_range = domain[:, 0].unique(sorted=True).numpy()
        x2_range = domain[:, 1].unique(sorted=True).numpy()
        domain = domain.numpy()
        x1_len = len(x1_range)
        x2_len = len(x2_range)
        x1_idxs = np.round(np.linspace(0, x1_len, 5)[1:-1]).astype(int)
        x2_idxs = np.round(np.linspace(0, x2_len, 4)[1:-1]).astype(int)
        inputs = [[x1_range[x1_idx], x2_range[x2_idx]] for x2_idx, x1_idx in product(x2_idxs, x1_idxs)]
    elif input_dim == 1:
        domain = domain.numpy()
        x_len = len(domain)
        x_idxs = np.round(np.linspace(0, x_len, 8)[1:-1]).astype(int)
        inputs = domain[x_idxs]
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    chains_idx = np.zeros(6, dtype=int)
    for x_idx, x in enumerate(inputs):
        chains_idx[x_idx] = int(np.argwhere(np.all(domain == x, axis=1)).squeeze())
    chains = preds[:, chains_idx].squeeze()  # network output traces

    # Now generate ACF plots for each slice
    fig, ax = plt.subplots(nrows=3, ncols=2)
    for rr in range(3):
        for vv in range(2):
            # Compute ESS on samples used for prediction
            chain = chains[:, 2 * rr + vv].squeeze()
            max_lag = max_corr_lag(chain)
            eff_ss = round(ess(chain, lag=max_lag))
            ess_str = 'ESS = {},'.format(eff_ss)

            # Determine x coordinate
            if input_dim == 2:
                x_coord = tuple(np.round(domain[chains_idx[2 * rr + vv], :], 2))
            else:
                x_coord = np.round(domain[chains_idx[2 * rr + vv]], 2).item()
            x_str = r'$\mathbf{{x}}$ = {}'.format(x_coord)

            # Compute Rhat on samples used for prediction, and add to panel along with other info
            rhat = compute_rhat(chain.reshape(-1, 1), n_chains=n_chains).item()
            rhat_str = r' $\widehat{{R}}$ = {:.4f}'.format(rhat)

            chain_acf = acf(chain, lag=max_lag, return_all=True, step=acf_step)
            markerline, stemline, baseline = ax[rr, vv].stem(np.arange(0, max_lag + 1), chain_acf)
            plt.setp(stemline, lw=1)
            plt.setp(markerline, ms=3)
            plt.setp(baseline, lw=1, c='k')
            conf_lim = 1.96 / np.sqrt(n_samples_kept)
            ax[rr, vv].hlines(y=[conf_lim], xmin=0, xmax=max_lag, linestyles=':', colors='k')
            ax[rr, vv].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[rr, vv].set_title(x_str + '\n' + ess_str + rhat_str)

    # Get the bounding boxes of the axes including text decorations
    fig.tight_layout()
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

def plot_cov_heatmap(covs, titles, domain, vmin = None, vmax = None):
    """
    Panelled covariance heatmaps for BNN and GP.

    :param covs: list, contains covariance matrices
    :param titles: list, contains strings with titles for corresponding cov matrices
    :param domain: torch.Tensor, array of all test inputs
    """
    domain = domain.detach().cpu()
    n_heatmaps = len(covs)
    input_dim = domain.shape[1]
    if input_dim == 2:
        x1_range = domain[:, 0].unique(sorted=True).numpy()
        x2_range = domain[:, 1].unique(sorted=True).numpy()
    elif input_dim == 1:
        x1_range = x2_range = domain.numpy()
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    if n_heatmaps > 4:
        n_rows = 2
        n_cols = n_heatmaps // 2
        if n_heatmaps / 2 != n_cols:
            raise Exception('If more than 4 heatmaps are plotted, must be an even number')
        figsize = (16, 12)
    else:
        n_rows = 1
        n_cols = n_heatmaps
        figsize = (16, 8)

    fig = plt.figure(figsize=figsize)
    plt.rc('font', size=22)
    gs = GridSpec(nrows=n_rows, ncols=n_cols, figure=fig)
    ax = np.empty((n_rows, n_cols), dtype=object)
    for rr, cc in product(range(n_rows), range(n_cols)):
        ax[rr, cc] = fig.add_subplot(gs[rr, cc])
    x1labs = np.round(x1_range, 1).flatten()
    x2labs = np.round(x2_range, 1).flatten()
    cov_df = np.empty((n_rows, n_cols), dtype=object)
    for rr in range(n_rows):
        for cc in range(n_cols):
            cov_df[rr, cc] = pd.DataFrame(covs[3*rr + cc], index=x2labs, columns=x1labs)

    if vmin is None:
        vmin = min([cov_df[rr, cc].values.min() for rr, cc in product(range(n_rows), range(n_cols))])
    if vmax is None:
        vmax = max([cov_df[rr, cc].values.max() for rr, cc in product(range(n_rows), range(n_cols))])
    yticks = cov_df[0, 0].index.values.round(2)
    ytick_idx = np.round(np.linspace(0, len(yticks) - 1, 8)).astype(int)
    xticks = cov_df[0, 0].columns.values.round(2)
    xtick_idx = np.round(np.linspace(0, len(xticks) - 1, 8)).astype(int)

    for rr in range(n_rows):
        for cc in range(n_cols):
            hm = sb.heatmap(cov_df[rr, cc], square=True, cbar=False, ax=ax[rr, cc],
                            vmin=vmin, vmax=vmax, yticklabels=False, xticklabels=False)
            hm.set_xticks(xtick_idx)
            hm.set_yticks(ytick_idx)
            if rr == n_rows - 1:
                hm.set_xticklabels(xticks[xtick_idx])
            if cc == 0:
                hm.set_yticklabels(yticks[ytick_idx])
            ax[rr, cc].set_title(titles[3*rr + cc])
            ax[rr, cc].xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=8, prune='upper'))
    fig.colorbar(mappable=ax[0, 0].collections[0], ax=ax, location='bottom', shrink=0.6)

def plot_cov_nonstat(cov, domain, cov_min=None, cov_max=None):
    """
    Panelled covariance heatmaps for BNN and GP (nonstationary version).

    :param cov: np.ndarray, covariance matrix between all inputs
    :param domain: torch.Tensor, array of all test inputs
    :param cov_min: float (optional), minimal covariance value
    :param cov_max: float (optional), maximal covariance value
    """
    domain = domain.float().detach().cpu()
    x1_range = domain[:, 0].unique(sorted=True).numpy().squeeze()
    x2_range = domain[:, 1].unique(sorted=True).numpy().squeeze()
    domain = domain.numpy()
    x1_len = len(x1_range)
    x2_len = len(x2_range)
    x1_idxs = np.round(np.linspace(0, x1_len, 6)[1:-1])
    x2_idxs = np.round(np.linspace(0, x2_len, 6)[1:-1])

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(nrows=4, ncols=4, figure=fig)
    ax = np.empty((4, 4), dtype=object)
    cov_df = ax.copy()
    titles = ax.copy()

    # Set values for each plot in grid
    for i, j in product(range(4), range(4)):
        ax[i, j] = fig.add_subplot(gs[i, j])
        x1_idx = x1_idxs[j]
        x2_idx = x2_idxs[3 - i]
        x1 = x1_range[int(x1_idx)]
        x2 = x2_range[int(x2_idx)]
        x_idx = np.argwhere(np.all(domain == [x1, x2], axis=1)).squeeze()
        cov_mx_flat = (cov[x_idx, :].reshape(1, -1) + cov[:, x_idx].reshape(1, -1)) / 2  # take average
        cov_mx = np.flip(cov_mx_flat.reshape(x2_len, x1_len), axis=0)
        cov_df[3-i, j] = pd.DataFrame(cov_mx, index=np.flip(x2_range), columns=x1_range)
        cov_df[3-i, j].index = cov_df[3-i, j].index.to_series().round(1)  # round x2 labels
        cov_df[3-i, j].columns = cov_df[3-i, j].columns.to_series().round(1)  # round x1 labels
        titles[3-i, j] = str(tuple(np.round([x1, x2], 2)))  # store subplot title with coordinates

    # Obtain min/max covariance values
    if cov_min is None:
        vmin = min([cov_df[rr, cc].values.min() for rr, cc in product(range(4), range(4))])
    else:
        vmin = cov_min
    if cov_max is None:
        vmax = max([cov_df[rr, cc].values.max() for rr, cc in product(range(4), range(4))])
    else:
        vmax = cov_max

    # Generate grid of plots
    for i, j in product(range(4), range(4)):
        x1_idx = x1_idxs[j]
        x2_idx = np.flip(x2_idxs)[3-i]  # compare with the index definition for cov_df
        sb.heatmap(cov_df[3-i, j], cbar=False, ax=ax[i, j], vmin=vmin, vmax=vmax, xticklabels=7, yticklabels=7)
        ax[i, j].plot(x1_idx, x2_idx, 'k+', ms=20)  # add point marker (vertical x2 index is opposite of x2 value)
        ax[i, j].set_title(titles[3-i, j])  # add subplot title with coordinates
        if j > 0:
            ax[i, j].tick_params(labelleft=False)
        if i < 3:
            ax[i, j].tick_params(labelbottom=False)
    fig.colorbar(mappable=ax[0, 0].collections[0], ax=ax[:, -1], location='right', shrink=0.6)

    # Return min/max covariances for use in another plot (to match colour scales)
    return vmin, vmax

def plot_cov_nonstat_diff(cov, gp_cov, domain):
    """
    Panelled heatmaps of covariance differences between BNN and GP (nonstationary version).

    :param cov: np.ndarray, BNN covariance matrix between all inputs
    :param gp_cov: np.ndarray, GP covariance matrix between all inputs
    :param domain: torch.Tensor, array of all test inputs
    """
    domain = domain.float().detach().cpu()
    x1_range = domain[:, 0].unique(sorted=True).numpy().squeeze()
    x2_range = domain[:, 1].unique(sorted=True).numpy().squeeze()
    domain = domain.numpy()
    x1_len = len(x1_range)
    x2_len = len(x2_range)
    x1_idxs = np.round(np.linspace(0, x1_len, 6)[1:-1])
    x2_idxs = np.round(np.linspace(0, x2_len, 6)[1:-1])

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(nrows=4, ncols=4, figure=fig)
    ax = np.empty((4, 4), dtype=object)
    diff_df = ax.copy()
    titles = ax.copy()

    # Set values for each plot in grid
    for i, j in product(range(4), range(4)):
        ax[i, j] = fig.add_subplot(gs[i, j])
        x1_idx = x1_idxs[j]
        x2_idx = x2_idxs[3 - i]
        x1 = x1_range[int(x1_idx)]
        x2 = x2_range[int(x2_idx)]
        x_idx = np.argwhere(np.all(domain == [x1, x2], axis=1)).squeeze()
        cov_mx_flat = (cov[x_idx, :].reshape(1, -1) + cov[:, x_idx].reshape(1, -1)) / 2  # take average
        gp_cov_mx_flat = (gp_cov[x_idx, :].reshape(1, -1) + gp_cov[:, x_idx].reshape(1, -1)) / 2  # take average
        cov_mx = np.flip(cov_mx_flat.reshape(x2_len, x1_len), axis=0)
        gp_cov_mx = np.flip(gp_cov_mx_flat.reshape(x1_len, x1_len), axis=0)
        diff_df[3-i, j] = pd.DataFrame(gp_cov_mx - cov_mx, index=np.flip(x2_range), columns=x1_range)
        diff_df[3-i, j].index = diff_df[3-i, j].index.to_series().round(1)  # round x2 labels
        diff_df[3-i, j].columns = diff_df[3-i, j].columns.to_series().round(1)  # round x1 labels
        titles[3-i, j] = str(tuple(np.round([x1, x2], 2)))  # store subplot title with coordinates
    vmin = min([diff_df[rr, cc].values.min() for rr, cc in product(range(4), range(4))])  # min value across all panes
    vmax = max([diff_df[rr, cc].values.max() for rr, cc in product(range(4), range(4))])  # max value across all panes

    # Generate grid of plots
    for i, j in product(range(4), range(4)):
        x1_idx = x1_idxs[j]
        x2_idx = np.flip(x2_idxs)[3-i]  # compare with the index definition for cov_df
        sb.heatmap(diff_df[3-i, j], xticklabels=7, yticklabels=7, cbar=False, cmap='seismic', center=0,
                   ax=ax[i, j], vmin=vmin, vmax=vmax)  # heatmap of covariance differences
        ax[i, j].plot(x1_idx, x2_idx, 'k+', ms=20)  # add point marker (vertical x2 index is opposite of x2 value)
        ax[i, j].set_title(titles[3-i, j])  # add subplot title with coordinates
        if j > 0:
            ax[i, j].tick_params(labelleft=False)
        if i < 3:
            ax[i, j].tick_params(labelbottom=False)
    fig.colorbar(mappable=ax[0, 0].collections[0], ax=ax[:, -1], location='right', shrink=0.6)

def plot_cov_contours(cov1, domain, level, cov2=None, latent=None, perc_of_max=True):
    """
    Plot covariance contours for uniformly chosen points, for BNN and GP.

    :param cov1: np.ndarray, covariance matrix between all inputs (GP)
    :param domain: torch.Tensor, array of all test inputs
    :param level: float, plotted contour level as proportion of maximum covariance
    :param cov2: (optional) np.ndarray, second covariance matrix to plot (BNN)
    :param latent: (optional) np.ndarray, shape (n_test_v, n_test_h), latent function onto which to overlay contours
    :param perc_of_max: bool, specify whether plotted contours are 100*level % of maximum covariance, or fixed
    """
    domain = domain.detach().cpu()
    x1_range = domain[:, 0].unique(sorted=True).numpy()
    x2_range = domain[:, 1].unique(sorted=True).numpy()
    domain = domain.numpy()
    x1_len = len(x1_range)
    x2_len = len(x2_range)
    x1_idxs = np.round(np.linspace(0, x1_len, 8)[1:-1])
    x2_idxs = np.round(np.linspace(0, x2_len, 8)[1:-1])
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Plot underlying latent function
    plt.figure()
    if latent is not None:
        plt.contourf(X1, X2, latent, levels=256, cmap='Spectral_r', origin='lower')
        plt.axis('scaled')
        plt.colorbar(pad=0.01)

    # Plot each contour and corresponding point
    ax = plt.gca()
    if cov2 is not None:
        # Determine suitable contour levels
        lvl1_ = []
        lvl2_ = []
        for x1_idx, x2_idx in product(x1_idxs, x2_idxs):
            x1 = x1_range[int(x1_idx)]
            x2 = x2_range[int(x2_idx)]
            x_idx = np.argwhere(np.all(domain == [x1, x2], axis=1)).squeeze()
            cov_mx_flat1 = (cov1[x_idx, :].reshape(1, -1) + cov1[:, x_idx].reshape(1, -1)) / 2  # take average (GP)
            cov_mx_flat2 = (cov2[x_idx, :].reshape(1, -1) + cov2[:, x_idx].reshape(1, -1)) / 2  # ditto (BNN)
            lvl1_.append(np.max(cov_mx_flat1))
            lvl2_.append(np.max(cov_mx_flat2))

        # Select either 100*level % of max cov, or fixed level contours
        if perc_of_max:
            lvl1 = round(level * min(lvl1_), 3)
            lvl2 = round(level * min(lvl2_), 3)
        else:
            lvl1 = lvl2 = round(level, 3)

        # Now plot covariance contours, marking centre points
        for x1_idx, x2_idx in product(x1_idxs, x2_idxs):
            x1 = x1_range[int(x1_idx)]
            x2 = x2_range[int(x2_idx)]
            x_idx = np.argwhere(np.all(domain == [x1, x2], axis=1)).squeeze()
            cov_mx_flat1 = (cov1[x_idx, :].reshape(1, -1) + cov1[:, x_idx].reshape(1, -1)) / 2  # take average (GP)
            cov_mx_flat2 = (cov2[x_idx, :].reshape(1, -1) + cov2[:, x_idx].reshape(1, -1)) / 2  # ditto (BNN)
            cov_mx1 = cov_mx_flat1.reshape(x2_len, x1_len)  # reshape into cov matrix (GP)
            cov_mx2 = cov_mx_flat2.reshape(x2_len, x1_len)  # ditto (BNN)
            ct1 = ax.contour(X1, X2, cov_mx1, [lvl1], colors='xkcd:blue')  # add contour line (GP)
            ct2 = ax.contour(X1, X2, cov_mx2, [lvl2], colors='xkcd:red')  # ditto (BNN)
            plt.setp(ct1.collections[0], label='_nolegend_')
            plt.setp(ct2.collections[0], label='_nolegend_')
            plt.plot(x1, x2, 'k+', label='_nolegend_')  # add point marker
        legend_elements = [Line2D([0], [0], ls='-', lw=1, color='xkcd:blue'),
                           Line2D([0], [0], ls='-', lw=1, color='xkcd:red')]
        if perc_of_max:
            plt.legend(handles=legend_elements,
                       labels=['GP (Level-{:.2f})'.format(lvl1), 'BNN (Level-{:.2f})'.format(lvl2)],
                       loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
            plt.title(r'{:d}%-of-Maximum Covariance Contours'.format(round(100 * level)))
        else:
            plt.legend(handles=legend_elements, labels=['GP', 'BNN'],
                       loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
            plt.title(r'Level-{:.2f} Covariance Contours'.format(level))
    else:
        for x1_idx, x2_idx in product(x1_idxs, x2_idxs):
            x1 = x1_range[int(x1_idx)]
            x2 = x2_range[int(x2_idx)]
            x_idx = np.argwhere(np.all(domain == [x1, x2], axis=1)).squeeze()
            cov_mx_flat = (cov1[x_idx, :].reshape(1, -1) + cov1[:, x_idx].reshape(1, -1)) / 2  # take average (GP)
            cov_mx = cov_mx_flat.reshape(x2_len, x1_len)  # reshape into cov matrix (GP)
            ax.contour(X1, X2, cov_mx, [level], colors='xkcd:blue')  # add contour line (GP)
            plt.plot(x1, x2, 'k+')  # add point marker
    plt.axis('scaled')

def fmt(x, pos):
    """
    Scientific notation formatting for the number x

    :param x: float, tick value
    :param pos: tick position
    :return: string, tick label
    """
    mantissa, power = '{:.2e}'.format(x).split('e')
    power = int(power)
    if power == 0:
        return str(mantissa)
    else:
        return '{}e{}'.format(mantissa, power)

def plot_mean_sd(mean_grid, sd_grid, domain, obs=None, sd_range=None, mean_range=None):
    """
    Panelled plots of sample means and standard deviations (suitable for samples with 2D inputs).

    :param mean_grid: np.ndarray, shape (n_test_v, n_test_h), grid of sample means
    :param sd_grid: np.ndarray, shape (n_test_v, n_test_h), grid of sample std deviations
    :param domain: torch.Tensor, array of all test inputs
    :param obs: (optional) np.ndarray, size (n_train, 2), columns contain x1 and x2 coordinates of observations
    :param sd_range: (optional) list [float], specify [vmin, vmax] for std deviation plot
    :param mean_range: (optional) list [float], specify [vmin, vmax] for mean plot
    """
    domain = domain.detach().cpu()
    x1_range = domain[:, 0].unique(sorted=True).numpy()
    x2_range = domain[:, 1].unique(sorted=True).numpy()

    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[10, 3, 10])
    ax = np.empty(2, dtype=object)
    for cc in range(2):
        ax[cc] = fig.add_subplot(gs[2*cc])
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Customise SD and mean value ranges
    if sd_range is not None:
        sd_levels = np.linspace(sd_range[0], sd_range[1], 256)
    else:
        sd_levels = 256
    if mean_range is not None:
        mean_levels = np.linspace(mean_range[0], mean_range[1], 256)
    else:
        mean_levels = 256

    # Generate panelled plots and append colourbar
    if (mean_range is not None) & (sd_range is not None):
        ims = [ax[0].contourf(X1, X2, mean_grid, levels=mean_levels, cmap='Spectral', origin='lower',
                              vmin=mean_range[0], vmax=mean_range[1]),
               ax[1].contourf(X1, X2, sd_grid, levels=sd_levels, cmap='BrBG', origin='lower',
                              vmin=sd_range[0], vmax=sd_range[1])]
    else:
        ims = [ax[0].contourf(X1, X2, mean_grid, levels=mean_levels, cmap='Spectral', origin='lower'),
               ax[1].contourf(X1, X2, sd_grid, levels=sd_levels, cmap='BrBG', origin='lower')]
    #titles = ['Prediction Mean', 'Prediction Standard Error']
    titles = ['', '']
    divider = []
    cax = []
    for cc in range(2):
        divider.append(make_axes_locatable(ax[cc]))
        cax.append(divider[cc].append_axes("right", size="5%", pad=0.15))
        ax[cc].set_title(titles[cc])

        # Plot observations (if specified)
        if obs is not None:
            ax[cc].scatter(obs[:, 0], obs[:, 1], marker='o', color='k', s=10)

        # Add colourbar with adjusted range
        if (sd_range is not None) & (cc == 1):
            fig.colorbar(mappable=ims[cc], cax=cax[cc], boundaries=np.linspace(sd_range[0], sd_range[1], 256),
                         format=FuncFormatter(fmt))
        elif (mean_range is not None) & (cc == 0):
            fig.colorbar(mappable=ims[cc], cax=cax[cc], boundaries=np.linspace(mean_range[0], mean_range[1], 256),
                         format=FuncFormatter(fmt))
        else:
            fig.colorbar(mappable=ims[cc], cax=cax[cc], format=FuncFormatter(fmt))

def plot_samples_2d(samples, extent, obs=None, figsize=(14,13), 
                    nrow = 4, ncol = 4, vmin = None, vmax = None):
    """
    Panelled plots of 2D BNN/GP samples (4-by-4 grid of panels).

    :param samples: np.ndarray, shape (n_samples, n_test_v, n_test_h), contains samples for plotting
    :param extent: float, or 2-tuple (float), or 4-tuple (float), specifying grid limits; (left, right, bottom, top)
    :param obs: (optional) np.ndarray, size (n_train, 2), columns contain x1 and x2 coordinates of observations
    :param figsize: (optional) tuple 2*(float), specify the (width, height) of the figure
    """
    if isinstance(extent, (tuple, list)):
        if len(extent) == 2:
            extent *= 2
    elif isinstance(extent, (float, int)):
        extent = [-abs(extent), abs(extent)] * 2
    else:
        raise TypeError
    assert isinstance(extent, list)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=nrow, ncols=ncol, figure=fig)
    ax = np.empty((nrow, ncol), dtype=object)
    for i, j in product(range(nrow), range(ncol)):
        ax[i, j] = fig.add_subplot(gs[i, j])

    samples_it = reversed(samples)  # turn array into iterator (along axis 0)
    ims = [ax[i, j].imshow(next(samples_it), cmap='Spectral_r',
                            extent=extent, origin='lower', 
                            aspect='equal', vmin = vmin, vmax = vmax)
           for i, j in product(range(nrow), range(ncol))]

    fig.colorbar(mappable=ims[0], ax=ax[:, -1], location='right', shrink=0.6)

    for i, j in product(range(nrow), range(ncol)):
        if obs is not None:
            ax[i, j].scatter(obs[:, 0], obs[:, 1], marker='o', color='k', s=10)
        if i < nrow - 1:
            ax[i, j].tick_params(labelbottom=False)  # only show x ticks for bottom row
        if j > 0:
            ax[i, j].tick_params(labelleft=False)  # only show y ticks for leftmost column
        ax[i, j].xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=4, prune='upper'))
        ax[i, j].yaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=4, prune='upper'))

def plot_bnn_grid(bnn_grid, domain, type, obs=None, bnn_idxs=None, figsize=(15,14)):
    """
    Panelled plots of sample means/SDs for each trained stationary BNN (for 2D nonstationary case).

    :param bnn_grid: np.ndarray, shape (grid_size, n_test_v, n_test_h), array of sample mean/SD grids
    :param domain: torch.Tensor, array of all test inputs
    :param bnn_locations: np.ndarray, shape (grid_size, input_dim), trained BNN locations
    :param type: str, either `mean`, `sd`, or `samples`, specify type of values to plot
    :param obs: (optional) np.ndarray, size (n_train, input_dim), columns contain x_i coordinates of observations
    :param bnn_idxs: (optional) list [int], indices of trained BNN locations
    :param figsize: (optional)
    """
    if type == 'mean':
        cmap = 'Reds'
    elif type == 'sd':
        cmap = 'Blues'
    elif type == 'samples':
        cmap = 'Spectral_r'
    else:
        raise Exception("Invalid type: require `mean`, `sd`, or `samples`")
    
    domain = domain.detach().cpu()
    x1_range = domain[:, 0].unique(sorted=True).numpy()
    x2_range = domain[:, 1].unique(sorted=True).numpy()
    grid_size = bnn_grid.shape[0]
    grid_len = int(np.sqrt(grid_size))

    # Extract BNN coordinates (take into account the locations are a regular grid, so each x1 pairs with each x2)
    if bnn_idxs is not None:
        bnn_loc = domain[bnn_idxs, :]
        x1_bnn = np.sort(np.unique(domain[bnn_idxs, 0]))  # sort in ascending order
        x2_bnn = np.sort(np.unique(domain[bnn_idxs, 1]))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=grid_len, ncols=grid_len, figure=fig)
    ax = np.empty((grid_len, grid_len), dtype=object)
    im = ax.copy()
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Generate grid of plots
    grid_max = grid_len - 1
    for i, j in product(range(grid_len), range(grid_len)):  # j index ticks first (corresponding to x1)
        ax[grid_max - i, j] = fig.add_subplot(gs[grid_max - i, j])
        gg = j + i * grid_len
        im[grid_max - i, j] = \
            ax[grid_max - i, j].contourf(X1, X2, bnn_grid[gg, :, :], levels=256, cmap=cmap, origin='lower')
        # This gives the right BNN since each tick in j corresponds to a tick in x1
        if obs is not None:
            ax[grid_max - i, j].scatter(obs[:, 0], obs[:, 1], marker='o', color='k', s=10)
        if bnn_idxs is not None:
            x1 = x1_bnn[j]
            x2 = x2_bnn[i]
            ax[grid_max - i, j].plot(x1, x2, color='k', marker='+', ms=40)
            ax[grid_max - i, j].set_title(str(tuple(np.round([x1, x2], 2))))
        if j > 0:
            ax[grid_max - i, j].tick_params(labelleft=False)
        if i > 0:
            ax[grid_max - i, j].tick_params(labelbottom=False)
        ax[grid_max - i, j].xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=4, prune='upper'))
    fig.colorbar(mappable=im[0, 0], ax=ax[:, -1], location='right', shrink=0.6)  # add colourbar

def plot_cov_diff(covs, gp_cov, titles, domain):
    """
    Plot differences of covariance heatmaps (to compare BNN covariance structures relative to GP).

    :param covs: list [np.ndarray], contains BNN covariance matrices
    :param gp_cov: np.ndarray, GP covariance matrix
    :param titles: list, contains strings with titles for corresponding cov matrices
    :param domain: torch.Tensor, array of all test inputs
    """
    domain = domain.detach().cpu()
    n_heatmaps = len(covs)
    input_dim = domain.shape[1]
    if input_dim == 2:
        x1_range = domain[:, 0].unique(sorted=True).numpy()
        x2_range = domain[:, 1].unique(sorted=True).numpy()
    elif input_dim == 1:
        x1_range = x2_range = domain.numpy()
    else:
        raise NotImplementedError('Only implemented for n=1 and n=2 input dimensions')

    # Compute differences of covariance matrices
    cov_diffs = []
    for cov in covs:
        cov_diffs.append(gp_cov - cov)

    fig = plt.figure(figsize=(16, 8))
    plt.rc('font', size=22)
    gs = GridSpec(nrows=1, ncols=n_heatmaps, figure=fig)
    ax = np.empty(n_heatmaps, dtype=object)
    for hh in range(n_heatmaps):
        ax[hh] = fig.add_subplot(gs[hh])
    x1labs = np.round(x1_range, 1).flatten()
    x2labs = np.round(x2_range, 1).flatten()
    cov_df = np.empty(n_heatmaps, dtype=object)
    for hh in range(n_heatmaps):
        cov_df[hh] = pd.DataFrame(cov_diffs[hh], index=x2labs, columns=x1labs)

    vmin = min([cov_df[hh].values.min() for hh in range(n_heatmaps)])
    vmax = max([cov_df[hh].values.max() for hh in range(n_heatmaps)])
    yticks = cov_df[0].index.values.round(2)
    ytick_idx = np.round(np.linspace(0, len(yticks) - 1, 8)).astype(int)
    xticks = cov_df[0].columns.values.round(2)
    xtick_idx = np.round(np.linspace(0, len(xticks) - 1, 8)).astype(int)

    for hh in range(n_heatmaps):
        hm = sb.heatmap(cov_df[hh], square=True, cbar=False, ax=ax[hh], cmap='seismic',
                        vmin=vmin, vmax=vmax, center=0, yticklabels=False, xticklabels=False)
        hm.set_xticks(xtick_idx)
        hm.set_yticks(ytick_idx)
        hm.set_xticklabels(xticks[xtick_idx])
        if hh == 0:
            hm.set_yticklabels(yticks[ytick_idx])
        ax[hh].set_title(titles[hh])
    fig.colorbar(mappable=ax[0].collections[0], ax=ax, location='bottom', shrink=0.6)

def plot_sst(t, sst_data, lat, lon, show_grid=True):
    """
    Function for generating plot of SST data at given time, optionally showing the grid of 64-by-64 regions used.

    :param t: int, which time step to extract data at
    :param sst_data: np.ndarray, contains SST data, dimensions [time, lat, long]
    :param lat: np.ndarray or netcdf_variable, contains latitudinal coordinates for SST data
    :param lon: np.ndarray or netcdf_variable, contains longitudinal coordinates for SST data
    :param show_grid: bool, specify if the grid of 64-by-64 regions should be displayed or not
    """
    sst_data = np.squeeze(sst_data)  # squeezing changes dimensions to [time, lat, long]
    plt.figure(figsize=[1.5 * x for x in plt.rcParams["figure.figsize"]])
    #plt.figure(figsize=(14, 12))
    X1, X2 = np.meshgrid(list(lon), list(lat), indexing='xy')  # (x, y) = (lon, lat)
    plt.contourf(X1, X2, sst_data[t, :, :], levels=256, cmap='Spectral_r', origin='lower', aspect='equal')

    if show_grid:
        # Plot grid of all 64-by-64 regions
        n_lon = len(list(lon))
        n_lat = len(list(lat))
        lat_vals = lat[slice(0, n_lat, 64)]
        lon_vals = lon[slice(0, n_lon, 64)]
        plt.hlines(y=lat_vals, xmin=min(lon), xmax=max(lon), colors='k', linestyles='-', linewidths=0.5)
        plt.vlines(x=lon_vals, ymin=min(lat), ymax=max(lat), colors='k', linestyles='-', linewidths=0.5)

        # Obtain vertical grid line indices for SST region
        lon_vals0 = lon[slice(0, n_lon - 64, 64)]
        lon_vals1 = lon[slice(64, n_lon - 2 * 64, 64)]
        lon_vals2 = lon[slice(4 * 64, n_lon, 64)]
        lon_vals3 = lon[slice(4 * 64, n_lon, 64)]
        lon_vals4 = lon[slice(3 * 64, n_lon, 64)]

        # Vertical grid lines
        plt.vlines(x=lon_vals0, ymin=lat[0], ymax=lat[64], colors='k', linestyles='-', linewidths=3)
        plt.vlines(x=lon_vals1, ymin=lat[64], ymax=lat[2 * 64], colors='k', linestyles='-', linewidths=3)
        plt.vlines(x=lon_vals2, ymin=lat[2 * 64], ymax=lat[3 * 64], colors='k', linestyles='-', linewidths=3)
        plt.vlines(x=lon_vals3, ymin=lat[3 * 64], ymax=lat[4 * 64], colors='k', linestyles='-', linewidths=3)
        plt.vlines(x=lon_vals4, ymin=lat[4 * 64], ymax=lat[5 * 64], colors='k', linestyles='-', linewidths=3)

        # Horizontal grid lines
        plt.hlines(y=lat[0], xmin=min(lon_vals0), xmax=max(lon_vals0),
                   colors='k', linestyles='-', linewidths=3)
        plt.hlines(y=lat[64], xmin=min(min(lon_vals0), min(lon_vals1)), xmax=max(max(lon_vals0), max(lon_vals1)),
                   colors='k', linestyles='-', linewidths=3)
        plt.hlines(y=lat[2 * 64], xmin=min(min(lon_vals1), min(lon_vals2)), xmax=max(max(lon_vals1), max(lon_vals2)),
                   colors='k', linestyles='-', linewidths=3)
        plt.hlines(y=lat[3 * 64], xmin=min(min(lon_vals2), min(lon_vals3)), xmax=max(max(lon_vals2), max(lon_vals3)),
                   colors='k', linestyles='-', linewidths=3)
        plt.hlines(y=lat[4 * 64], xmin=min(min(lon_vals3), min(lon_vals4)), xmax=max(max(lon_vals3), max(lon_vals4)),
                   colors='k', linestyles='-', linewidths=3)
        plt.hlines(y=lat[5 * 64], xmin=min(lon_vals4), xmax=max(lon_vals4),
                   colors='k', linestyles='-', linewidths=3)

    # Add colourbar
    plt.axis('scaled')
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('Degrees (Celsius)', rotation=270, labelpad=24)

def plot_wasserstein_iterations(wdist_vals, indices, FIG_DIR):
    plt.figure()
    plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
    plt.ylabel(r'$W_1(p_{gp}, p_{nn})$')
    plt.xlabel('Iteration (Outer Loop)')
    plt.savefig(FIG_DIR + '/Wasserstein_steps.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
    plt.yscale('log')
    plt.ylabel(r'$W_1(p_{gp}, p_{nn})$')
    plt.xlabel('Iteration (Outer Loop)')
    plt.savefig(FIG_DIR + '/Wasserstein_steps_log.png', bbox_inches='tight')