import numpy as np
import math
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
from copy import deepcopy
from itertools import islice, product
from collections import ChainMap
import time

from ..samplers.adaptive_sghmc import AdaptiveSGHMC
from ..samplers.sghmc import SGHMC
from ..utils.util import inf_loop, prepare_device
from ..utils.normalisation import zscore_normalisation, zscore_unnormalisation
from ..bnn.layers.embedding_layer import EmbeddingLayer


class BayesNet:
    def __init__(self, net, likelihood, prior, sampling_method="adaptive_sghmc", n_gpu=0,
                 normalise_input=False, normalise_output=True):
        """
        Bayesian neural network that uses stochastic gradient MCMC to sample from the posterior.

        :param net: nn.Module, the base neural network (excluding embedding layer)
        :param likelihood: instance of LikelihoodModule, the module for the likelihood
        :param prior: instance of PriorModule, the module for the prior
        :param sampling_method: str, specify the sampling strategy
        :param n_gpu: int, number of GPUs to use for computation
        :param normalise_input: bool, specify whether inputs are normalised
        :param normalise_output: bool, specify whether outputs are normalised
        """
        self.net = net
        self.lik_module = likelihood
        self.prior_module = prior
        self.n_gpu = n_gpu
        self.do_normalise_input = normalise_input
        self.do_normalise_output = normalise_output

        # Initialise attributes for use with embedding layer
        self.domain = None
        self.embedding_layer = None
        self.rbf = None

        # Nonstationarity settings
        self.nonstationary = False
        self.bnn_idxs = [0]
        self.grid_size = None
        self.bnn_loc = None

        # MCMC sampling settings
        self.sampling_method = sampling_method
        self.step = 0
        self.sampler = None
        self.chain_count = 0  # keep track of how many chains have been sampled
        self.num_chains = None  # keep track of total number of chains
        self.sampled_weights = None
        self.pred_weights = None

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = prepare_device(self.n_gpu)
        self.net = self.net.to(self.device)
        if len(device_ids) > 1:
            self.net = torch.nn.DataParallel(net, device_ids=device_ids)
        self.prior_module = self.prior_module.to(self.device)

    def add_embedding_layer(self, input_dim, rbf_dim, domain, rbf_ls=1):
        """
        Add embedding layer, if it was used in the BNN prior.

        :param input_dim: int, number of input dimensions
        :param rbf_dim: int, number of RBFs (embedding layer width)
        :param domain: torch.Tensor, contains all test inputs in the rows
        :param rbf_ls: float, length-scale of the RBFs
        """
        self.domain = domain
        self.embedding_layer = EmbeddingLayer(input_dim=input_dim,
                                              output_dim=rbf_dim,
                                              domain=domain,
                                              rbf_ls=rbf_ls)
        self.rbf = self.embedding_layer(domain)

    def make_nonstationary(self, grid_height, grid_width):
        """
        Enable the posterior BNN to model nonstationarity, via Bayesian model averaging.

        :param grid_height: int, height of grid of spatial locations where SGHMC is applied
        :param grid_width: int, width of grid of spatial locations where SGHMC is applied
        """
        if self.domain is None:
            raise Exception('Instantiate embedding layer before incorporating nonstationarity.')
        self.nonstationary = True
        self.grid_size = grid_height * grid_width

        domain = self.domain.detach().cpu()
        x1_range = domain[:, 0].unique(sorted=True).numpy()
        x2_range = domain[:, 1].unique(sorted=True).numpy()
        x1_len = len(x1_range)
        x2_len = len(x2_range)

        # Obtain indices of the boundaries between S_i subregions
        x1_bdry = np.int_(np.round(np.linspace(0, x1_len, grid_width + 1)))
        x2_bdry = np.int_(np.round(np.linspace(0, x2_len, grid_height + 1)))

        # Obtain indices of the centroids in each S_i subregion
        x1_idxs = np.array([(x1_bdry[i] + x1_bdry[i + 1]) // 2 for i in range(grid_width)])
        x2_idxs = np.array([(x2_bdry[i] + x2_bdry[i + 1]) // 2 for i in range(grid_height)])

        # Obtain spatial coordinates corresponding to SGHMC locations
        x1_subset = x1_range[x1_idxs]
        x2_subset = x2_range[x2_idxs]
        x1_coords, x2_coords = np.meshgrid(x1_subset, x2_subset)
        test_subset = np.vstack((x1_coords.flatten(), x2_coords.flatten())).T  # 1st index (x1) changes fastest

        # Row indices of SGHMC locations in the tensor containing all test inputs
        domain = domain.numpy()
        self.bnn_idxs = [np.argwhere(np.all(domain == test_subset[rr, :], axis=1)).item()
                         for rr in range(test_subset.shape[0])]

    def _neg_log_joint(self, fx_batch, y_batch, n_train, test_input=None):
        """
        Calculate the model's negative log joint density (the potential energy function in SGHMC).

        :param fx_batch: torch.Tensor, network predictions
        :param y_batch: torch.Tensor, corresponding noisy targets (observations)
        :param n_train: int, size of training set
        :param test_input: int, specify row index of test input in the (n_test_h*n_test_v, input_dim) array
        :return: torch.Tensor, negative log joint density (both likelihood and prior)

        Note: the mini-batch gradient is computed as grad_prior + N/n sum_i grad_likelihood_xi, see eq. 5 in [1].
        Therefore, we divide the prior by the training set size N, since we will rescale the gradient by N again
        in the sampler (the value of N is precisely scale_grad in sampler_params).

        [1] Chen et al. 2014 (Stochastic gradient Hamiltonian Monte Carlo)
        """
        batch_size = y_batch.shape[0]  # number of training points in mini-batch
        likelihood = self.lik_module(fx_batch, y_batch)  # negative log likelihood
        prior = self.prior_module(self.net, test_input)  # negative log prior
        return likelihood / batch_size + prior / n_train

    def _neg_log_lik(self, fx_batch, y_batch):
        """
        Negative log likelihood.

        :param fx_batch: torch.Tensor, network predictions
        :param y_batch: torch.Tensor, corresponding noisy targets (observations)
        :return: torch.Tensor, negative log likelihood
        """
        batch_size = y_batch.shape[0]
        likelihood = self.lik_module(fx_batch, y_batch)
        return likelihood / batch_size

    def _neg_log_prior(self, n_train, test_input=None):
        """
        Negative log prior.

        :param n_train: int, size of training set
        :param test_input: int, specify row index of test input in the (n_test_h*n_test_v, input_dim) array
        :return: torch.Tensor, negative log prior
        """
        prior = self.prior_module(self.net, test_input)
        return prior / n_train

    def _initialise_sampler(self, n_train, lr=1e-2, mdecay=0.05, num_burn_in_steps=3000, epsilon=1e-10):
        """
        Initialise the stochastic gradient MCMC sampler.

        :param n_train: int, size of training set
        :param lr: float, learning rate
        :param mdecay: float, momentum decay
        :param num_burn_in_steps: int, number of burn-in steps to perform (given to the sampler if it supports special
            burn-in specific behaviour, such as that of adaptive SGHMC
        :param epsilon: float, small positive number added for numerical stability
        """
        dtype = np.float32  # originally np.float32
        self.sampler_params = {}

        self.sampler_params['scale_grad'] = dtype(n_train)  # multiplies the mini-batch gradient
        self.sampler_params['lr'] = dtype(lr)
        self.sampler_params['mdecay'] = dtype(mdecay)

        if self.sampling_method == "adaptive_sghmc":
            self.sampler_params['num_burn_in_steps'] = num_burn_in_steps
            self.sampler_params['epsilon'] = dtype(epsilon)
            self.sampler = AdaptiveSGHMC(self.net.parameters(), **self.sampler_params)
        elif self.sampling_method == "sghmc":
            self.sampler = SGHMC(self.net.parameters(), **self.sampler_params)

        # Note: net.parameters() is a generator which can be iterated through to obtain parameter tensors.

    def predict(self, x_test):
        """
        Predicts latent target values for the given test input(s).

        :param x_test: np.ndarray or torch.Tensor, shape (*, input_dim), the raw test input(s)
        :return: tuple, 2*(np.ndarray), predictions
        """
        n_test = x_test.shape[0]  # number of test inputs for prediction
        n_burn = self.len_sampled_chain - self.len_pred_chain  # number of burn-in samples; used later

        # Normalise the input data
        if isinstance(x_test, np.ndarray):
            x_test = torch.from_numpy(x_test).float().to(self.device)
        if self.embedding_layer is not None:
            if len(x_test.shape) == 1:
                x_test = x_test.unsqueeze(1)
            rbf_test = self.embedding_layer(x_test)
            if self.do_normalise_input:
                rbf_test = rbf_test.detach().cpu().numpy().squeeze()
                rbf_test, *_ = zscore_normalisation(rbf_test, self.input_mean, self.input_std)
            input_test = rbf_test
        else:
            input_test = x_test
        if isinstance(input_test, np.ndarray):
            input_test = torch.from_numpy(input_test)
        input_test = input_test.float().to(self.device)

        def network_predict(test_input, weights):
            """
            Function producing network predictions at given inputs, for specified BNN weights.

            :param test_input: torch.Tensor, inputs upon which to perform predictions
            :param weights: list, contains BNN state dicts with parameter tensors, for each MCMC step
            :return: np.ndarray, associated predictions, shape (*, output_dim)
            """
            with torch.no_grad():  # disable gradient calculations, to improve speed
                self.net.load_state_dict(weights)  # assign parameters to neural net
                preds = self.net(test_input).detach().cpu().numpy()  # return prediction as numpy array
                return preds.squeeze()

        # Obtain predictions for each test input in nonstationary (first) or stationary (second) case
        if self.nonstationary:
            bnn_x_test = x_test[self.bnn_idxs, :]  # trained BNN locations

            start_time = time.perf_counter()

            # Compute coefficients in convex combination
            bnn_x = torch.tile(bnn_x_test, (n_test, 1))  # behaves like np.tile
            pred_x = torch.repeat_interleave(x_test, repeats=self.grid_size, dim=0)  # behaves like np.repeat
            dist = torch.norm((bnn_x - pred_x), dim=1).squeeze()  # distances from BNN locations
            dist_bnn = torch.empty(n_test, self.grid_size)  # reshaped distances, (# test inputs, # trained BNNs)
            for g in range(self.grid_size):
                dist_bnn[:, g] = dist[g::self.grid_size]
            weights_unnorm = 1 / (1 + dist_bnn ** 2)  # shape (n_test, grid_size)
            weights_sum = torch.sum(weights_unnorm, dim=1).unsqueeze(1).tile(1, self.grid_size)
            weights_norm = weights_unnorm / weights_sum

            mid_time = time.perf_counter()

            # Obtain trained BNN predictions for each spatial input
            preds_all_bnn = np.empty((self.grid_size, self.len_sampled_chain * self.num_chains, n_test))
            for gg in range(self.grid_size):
                print('Obtaining predictions for grid point # {}/{}'.format(gg+1, self.grid_size))
                bnn_preds_array = np.array([network_predict(input_test, weights)
                                            for weights in self.sampled_weights[gg]])
                if gg == 0:
                    print('Storage array has shape {}'.format(preds_all_bnn.shape))
                print('BNN predictions array has shape {}'.format(bnn_preds_array.shape))
                preds_all_bnn[gg, :, :] = bnn_preds_array

            mid2_time = time.perf_counter()

            # Mix predictions based on density mixture weightings
            convex_cumul = np.cumsum(weights_norm, axis=1).detach().cpu().numpy()
            runif = np.random.random((n_test, self.len_sampled_chain * self.num_chains))  # random [0,1] number
            preds_all = np.empty((self.len_sampled_chain * self.num_chains, n_test))
            inserted_preds = np.zeros_like(preds_all)  # keep track of which predictions were sampled
            for ss in range(n_test):
                for gg in range(self.grid_size):
                    which_preds = (runif[ss, :] <= convex_cumul[ss, gg]) & (inserted_preds[:, ss] == 0)
                    preds_all[which_preds, ss] = preds_all_bnn[gg, which_preds, ss]
                    inserted_preds[which_preds, ss] += 1

            # Extract predictions corresponding to retained sampled weights
            preds = np.empty((self.len_pred_chain * self.num_chains, n_test))
            preds_bnn = np.empty((self.grid_size, self.len_pred_chain * self.num_chains, n_test))
            for cc in range(self.num_chains):
                preds[cc * self.len_pred_chain : (cc + 1) * self.len_pred_chain, :] \
                    = preds_all[n_burn + cc * self.len_sampled_chain : (cc + 1) * self.len_sampled_chain, :]
                preds_bnn[:, cc * self.len_pred_chain : (cc + 1) * self.len_pred_chain, :] \
                    = preds_all_bnn[:, n_burn + cc * self.len_sampled_chain : (cc + 1) * self.len_sampled_chain, :]

            end_time = time.perf_counter()
            print('Time: {:.4f}s, {:.4f}s, {:.4f}s'.format(mid_time-start_time, mid2_time-mid_time, end_time-mid2_time))
        else:
            # Make predictions for each set of sampled weights (locationally invariant)
            preds_all = np.array([network_predict(input_test, weights) for weights in self.sampled_weights])

            # Extract predictions corresponding to retained sampled weights
            preds = np.empty((self.len_pred_chain * self.num_chains, n_test))
            for cc in range(self.num_chains):
                preds[cc * self.len_pred_chain: (cc + 1) * self.len_pred_chain, :] \
                    = preds_all[n_burn + cc * self.len_sampled_chain: (cc + 1) * self.len_sampled_chain, :]

        # Unnormalise the output data
        if self.do_normalise_output:
            preds = zscore_unnormalisation(preds, self.y_mean, self.y_std)
            preds_all = zscore_unnormalisation(preds_all, self.y_mean, self.y_std)
            if self.nonstationary:
                preds_bnn = zscore_unnormalisation(preds_bnn, self.y_mean, self.y_std)
                preds_all_bnn = zscore_unnormalisation(preds_all_bnn, self.y_mean, self.y_std)

        if self.nonstationary:
            return preds, preds_all, preds_bnn, preds_all_bnn
        return preds, preds_all

    def _normalise_data(self, input_train, y_train):
        """
        Normalise the training data.

        :param input_train: np.ndarray, training set inputs (x values or RBF values)
        :param y_train: np.ndarray, training set outputs (noisy targets)
        :return: tuple, 2*(torch.Tensor), normalised inputs and outputs
        """
        if self.do_normalise_input:
            input_train_, self.input_mean, self.input_std = zscore_normalisation(input_train)
            input_train_ = torch.from_numpy(input_train_).float()
        else:
            input_train_ = torch.from_numpy(input_train).float()

        if self.do_normalise_output:
            y_train_, self.y_mean, self.y_std = zscore_normalisation(y_train)
            y_train_ = torch.from_numpy(y_train_).float()
        else:
            y_train_ = torch.from_numpy(y_train).float()

        return input_train_, y_train_

    def sample_multi_chains(self,
                            x_train,
                            y_train,
                            num_samples,
                            num_chains=1,
                            keep_every=100,
                            n_discarded=0,
                            num_burn_in_steps=3000,
                            lr=1e-2,
                            batch_size=32,
                            epsilon=1e-10,
                            mdecay=0.05,
                            print_every_n_samples=10,
                            resample_prior_every=1000):
        """
        Use multiple chains of sampling (for MCMC convergence diagnostics, need >= 4).

        :param x_train: np.ndarray, training inputs
        :param y_train: np.ndarray, training targets
        :param num_samples: int, number of MCMC samples for each parameter (after thinning)
        :param num_chains: int, number of chains
        :param keep_every: int, thinning interval
        :param n_discarded: int, number of first samples to discard
        :param num_burn_in_steps: int, number of burn-in steps to perform (passed to the sampler if it supports special
            burn-in specific behaviour, such as that of adaptive SGHMC)
        :param lr: float, learning rate
        :param batch_size: int, batch size
        :param epsilon: float, small positive number added for numerical stability
        :param mdecay: float, momentum decay
        :param print_every_n_samples: int, interval at which to print statistics of sampling process
        :param resample_prior_every: int, number of sampling steps before resampling std devs of prior (for GPi-H)
        """
        # Compile settings together for brevity
        sampling_configs = {
            "x_train": x_train, "y_train": y_train, "num_samples": num_samples, "keep_every": keep_every,
            "n_discarded": n_discarded, "num_burn_in_steps": num_burn_in_steps, "lr": lr, "batch_size": batch_size,
            "epsilon": epsilon, "mdecay": mdecay, "print_every_n_samples": print_every_n_samples,
            "resample_prior_every": resample_prior_every
        }

        # Pre-allocate lists to improve performance (append causes performance issues)
        self.len_sampled_chain = num_samples + n_discarded + num_burn_in_steps // keep_every
        self.len_pred_chain = num_samples
        self.num_chains = num_chains

        # Train BNN for first chain
        print("Chain: 1")
        self.train(**sampling_configs)

        # Train BNN for each subsequent chain
        for cc in range(1, num_chains):
            print("Chain: {}".format(cc + 1))
            self.chain_count += 1
            self.train(**sampling_configs)

        # Return lists containing sampled network parameters for all chains
        return self.sampled_weights, self.pred_weights

    def train(self,
              x_train,
              y_train,
              num_samples,
              keep_every=100,
              n_discarded=0,
              num_burn_in_steps=3000,
              lr=1e-2,
              batch_size=32,
              epsilon=1e-10,
              mdecay=0.05,
              print_every_n_samples=10,
              resample_prior_every=1000):
        """
        Train a BNN using a given dataset (one chain of sampling).

        :param x_train: np.ndarray, training inputs
        :param y_train: np.ndarray, training targets
        :param num_samples: int, number of MCMC samples for each parameter (after thinning)
        :param keep_every: int, thinning interval
        :param n_discarded: int, number of first samples to discard
        :param num_burn_in_steps: int, number of burn-in steps to perform (passed to the sampler if it supports special
            burn-in specific behaviour, such as that of adaptive SGHMC)
        :param lr: float, learning rate
        :param batch_size: int, batch size
        :param epsilon: float, small positive number added for numerical stability
        :param mdecay: float, momentum decay
        :param print_every_n_samples: int, interval at which to print statistics of sampling process
        :param resample_prior_every: int, number of sampling steps before resampling std devs of prior (for GPi-H)
        """
        n_discarded_all = n_discarded + num_burn_in_steps // keep_every
        n_train = x_train.shape[0]

        # Compute RBF evaluations on training test inputs
        if self.embedding_layer is not None:
            if len(x_train.shape) == 1:
                x_train = x_train.unsqueeze(1)
            if isinstance(x_train, np.ndarray):
                x_train = torch.from_numpy(x_train).float()
            input_train = self.embedding_layer(x_train)
        else:
            input_train = x_train

        # Prepare the training dataset (normalising if specified)
        if isinstance(input_train, torch.Tensor):
            input_train = input_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()
        input_train, y_train = input_train.squeeze(), y_train.squeeze()
        input_train_, y_train_ = self._normalise_data(input_train, y_train)

        # Initialise a data loader for training data (loops through training set batches infinitely)
        train_loader = inf_loop(data_utils.DataLoader(data_utils.TensorDataset(input_train_, y_train_),
                                                      batch_size=batch_size, shuffle=True))

        # Number of times batch generator is cycled through
        num_cycles = 1
        if self.nonstationary:
            num_cycles = self.grid_size

        # Initialise the training set batch generator, providing a mini-batch for each MCMC step
        num_steps = (num_samples + n_discarded) * keep_every + num_burn_in_steps
        batch_generator = islice(enumerate(train_loader), num_steps * num_cycles)

        # Note: islice has arguments (iterable,stop) or (iterable,start,stop[,step]); makes an iterator

        self.net.train()  # set to training mode

        # Carry out Hamiltonian dynamics on network parameters for num_steps iterations, for each spatial test input
        for iter, (input_batch, y_batch) in batch_generator:

            step = iter + 1  # step 1 corresponds to iteration 0
            bnn_iter = iter % num_steps
            bnn_step = bnn_iter + 1
            bnn_num = (step - bnn_step) // num_steps

            test_input = None
            if self.nonstationary:
                test_input = self.bnn_idxs[bnn_num]

            # Initialise the stochastic gradient MCMC sampler
            if bnn_step == 1:
                print('Initialising MCMC sampler for BNN # {}'.format(bnn_num+1))
                num_sampled_dict = 0  # count total number of network parameter dictionaries per BNN
                num_pred_dict = 0  # count number of network parameter dictionaries used for prediction
                sampled_dict_list = [None] * self.len_sampled_chain  # container to store state_dicts
                self.net.reset_parameters()
                self._initialise_sampler(n_train, lr, mdecay, num_burn_in_steps, epsilon)

            input_batch, y_batch = input_batch.to(self.device), y_batch.to(self.device)
            input_batch = input_batch.view(y_batch.shape[0], -1)  # batch of training set inputs
            y_batch = y_batch.view(-1, 1)  # batch of training set noisy targets (observations)
            fx_batch = self.net(input_batch).view(-1, 1)  # network predictions on the input batch

            self.step += 1  # number of MCMC steps

            # Compute negative log joint density (potential energy function)
            loss = self._neg_log_joint(fx_batch, y_batch, n_train, test_input)
            #loss_lik = self._neg_log_lik(fx_batch, y_batch)
            #loss_prior = self._neg_log_prior(n_train, test_input)

            self.sampler.zero_grad()  # set gradients to zero (to prevent their accumulative addition)
            loss.backward()  # populate mini-batch gradient with derivatives dU/dp (U is loss)
            #loss_lik.backward()
            #loss_prior.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100.)

            # Note: clip_grad_norm_ computes norm of all gradients together, as if concatenated into a vector,
            #       and modifies the gradients in-place if their norm exceeds the limit (indicated to be 100)

            # Update parameters by performing one SGHMC optimiser step
            self.sampler.step()

            # Resample variances of the prior every R steps (hierarchical prior only)
            if self.prior_module.hyperprior:
                if bnn_step % resample_prior_every == 0:
                    if self.nonstationary:
                        self.prior_module.resample(self.net, test_input)
                    else:
                        self.prior_module.resample(self.net)

            # Save the network parameter dictionaries into a ChainMap, INCLUDING the burn-in parameters
            if (bnn_step % keep_every == 0) or (bnn_step == num_steps):
                net_dict = deepcopy(self.net.state_dict())  # extract current parameters
                sampled_dict_list[num_sampled_dict] = net_dict
                num_sampled_dict += 1
                if bnn_step == num_steps:
                    # Spatially-varying BNN: sampled_weights is a ChainMap with state_dict lists for each grid site
                    if self.nonstationary:
                        # For first Markov chain
                        if self.chain_count == 0:
                            if self.sampled_weights is None:
                                self.sampled_weights = ChainMap({bnn_num: sampled_dict_list})
                            elif bnn_num in self.sampled_weights.keys():
                                self.sampled_weights[bnn_num] += sampled_dict_list
                            else:
                                self.sampled_weights = self.sampled_weights.new_child({bnn_num: sampled_dict_list})
                        # For subsequent Markov chains (append list of dictionaries, for this grid site)
                        else:
                            self.sampled_weights[bnn_num] += sampled_dict_list
                    # Spatially-invariant BNN: sampled_weights is a list containing state_dicts for each MCMC step
                    else:
                        # For first Markov chain
                        if self.sampled_weights is None:
                            self.sampled_weights = sampled_dict_list
                        # For subsequent Markov chains (append list of dictionaries)
                        else:
                            self.sampled_weights += sampled_dict_list

                # Save the parameter dictionaries, EXCLUDING the burn-in parameters (i.e. those used for predictions)
                if num_sampled_dict > n_discarded_all:
                    num_pred_dict += 1
                    if bnn_step == num_steps:
                        pred_dict_list = sampled_dict_list[-self.len_pred_chain:]  # exclude the burn-in parameters
                        if self.nonstationary:
                            if self.chain_count == 0:
                                if self.pred_weights is None:
                                    self.pred_weights = ChainMap({bnn_num: pred_dict_list})
                                elif bnn_num in self.pred_weights.keys():
                                    self.pred_weights[bnn_num] += pred_dict_list
                                else:
                                    self.pred_weights = self.pred_weights.new_child({bnn_num: pred_dict_list})
                            else:
                                self.pred_weights[bnn_num] += pred_dict_list
                        else:
                            if self.pred_weights is None:
                                self.pred_weights = pred_dict_list
                            else:
                                self.pred_weights += pred_dict_list

                    # Print feedback
                    if (num_sampled_dict % print_every_n_samples == 0) or (bnn_step == num_steps):
                        feedback_str = "Step # {:8d} : Input # {:5d}/{:d} : Sample # {:5d} : Retained # {:5d}"
                        if bnn_step == num_steps:
                            feedback_str += " (*)"
                        print(feedback_str.format(self.step, bnn_num+1, len(self.bnn_idxs),
                                                  num_sampled_dict, num_pred_dict))
