import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import os, sys
import pickle
import matplotlib as mpl
import json

CWD = sys.path[0]
OUT_DIR = os.path.join(CWD, "runs/Section2_BNN_degeneracy/output")
FIG_DIR = os.path.join(CWD, "runs/Section2_BNN_degeneracy/figures")

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


sys.path.append(os.path.dirname(CWD))

from bnn_spatial.bnn.nets import GaussianNet, BlankNet
from bnn_spatial.bnn.layers.embedding_layer import EmbeddingLayer
from bnn_spatial.utils import util
from bnn_spatial.gp.model import GP
from bnn_spatial.gp import kernels, base
from bnn_spatial.stage1.wasserstein_mapper import MapperWasserstein
from bnn_spatial.utils.rand_generators import GridGenerator
from bnn_spatial.utils.plotting import plot_spread, plot_param_traces, plot_output_traces, plot_output_hist, \
    plot_output_acf, plot_cov_heatmap, plot_lipschitz, plot_rbf, plot_cov_diff
from bnn_spatial.stage2.likelihoods import LikGaussian
from bnn_spatial.stage2.priors import FixedGaussianPrior, OptimGaussianPrior
from bnn_spatial.stage2.bayes_net import BayesNet
from bnn_spatial.metrics.sampling import compute_rhat

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # for handling OMP: Error 15 (2022/11/21)

#plt.rcParams['figure.figsize'] = [14, 7]
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['image.aspect'] = 'auto'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24

#mpl.use('TkAgg')
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # set device to GPU if available

# Set seed (make results replicable)
util.set_seed(2023)


"""
Fixed BNN prior (non-optimised)
"""

n_test = 256  # number of test inputs (must be a multiple of "width" when using embedding layer)
Xmargin = 4  # margin of domain, so that the domain is [-Xmargin : step_size : Xmargin]
Xtest = np.linspace(-Xmargin, Xmargin, n_test).reshape(-1, 1)
Xtest_tensor = torch.from_numpy(Xtest).to(device)
n_plot = 4000  # number of BNN prior and GP samples to use in plots

# Initialize fixed BNN

# Specify BNN structure

for nlayer in [1, 2, 4, 8]:

    hidden_dims = [40]*nlayer  # list of hidden layer dimensions
    depth = len(hidden_dims)
    embed_width = hidden_dims[0]
    transfer_fn = "tanh"  # activation function

    rbf_ls = 1

    std_bnn = GaussianNet(input_dim=1,  
                        output_dim=1,
                        hidden_dims=hidden_dims,
                        domain=Xtest_tensor,
                        activation_fn=transfer_fn,
                        prior_per='layer',
                        fit_means=False,
                        rbf_ls=rbf_ls,
                        init_std=1).to(device)

    # Perform simulations using 'n_plot' number of sampled weights/biases
    std_bnn_samples = std_bnn.sample_functions(Xtest_tensor.float(), n_plot).detach().cpu().numpy().squeeze()

    # Here save the data as JSON for future reload in R
    with open(OUT_DIR + '/std_prior_preds_' + str(nlayer) + '_layers.json', 'w') as outfile:
        json.dump(std_bnn_samples.transpose().tolist(), outfile)


    # Obtain network parameter values
    # W_list, b_list = std_bnn.network_parameters()
    # W_std, b_std = W_list[0], b_list[0]  # all values are identical
