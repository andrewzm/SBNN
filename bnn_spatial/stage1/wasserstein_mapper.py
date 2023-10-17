"""
Calibrate BNN prior to target GP prior (or any other stochastic process)
"""

import torch
import torch.nn as nn
import numpy as np
import itertools, os
from torch.utils.data import TensorDataset, DataLoader
from ..utils.util import prepare_device, ensure_dir


class LipschitzFunction(nn.Module):
    def __init__(self, dim):
        """
        Instantiate neural network representing the Lipschitz function.

        :param dim: int, dimension of network input
        """
        super(LipschitzFunction, self).__init__()

        # Two hidden layers with 200 units, and softplus activation
        self.lin1 = nn.Linear(dim, 200)
        self.relu1 = nn.Softplus()
        self.lin2 = nn.Linear(200, 200)
        self.relu2 = nn.Softplus()
        self.lin3 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Performs network computation performed at every call.

        :param x: torch.Tensor, network input
        :return: torch.Tensor, network output
        """
        x = x.float()
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        return x

    def reset_parameters(self):
        """
        Reset Lipschitz network parameters.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

class WassersteinDistance():
    def __init__(self, bnn, gp, lipschitz_f_dim, wasserstein_lr=0.02, starting_lr=0.001, device='cpu',
                 gpu_gp=True, save_memory=False, raw_data=False, continue_training=False,
                 lognormal=False):
        """
        Code for computing Lipschitz losses and Wasserstein-1 distances, and Lipschitz network optimisation.

        :param bnn: nn.Module, BNN prior
        :param gp: nn.Module (instance of Kern child), GP prior
        :param lipschitz_f_dim: int, size of measurement set
        :param wasserstein_lr: float, learning rate for optimiser
        :param starting_lr: float, learning rate for optimiser during the first outer iteration
        :param device: str, default device for computations
        :param gpu_gp: bool, specify whether to compute GP samples using GPU or not
        :param save_memory: bool, specify whether memory demands should be reduced at expense of slower computation
        :param raw_data: bool, specify whether raw data samples are being used as the target prior
        :param continue_training: bool, specify whether Lipschitz network is pretrained or not
        """
        self.bnn = bnn
        self.gp = gp
        self.device = device
        self.lipschitz_f_dim = lipschitz_f_dim
        self.gpu_gp = gpu_gp
        self.penalty_coeff = 10  # coefficient in Lipschitz penalty term
        self.save_memory = save_memory
        self.raw_data = raw_data
        self.wasserstein_lr = wasserstein_lr
        self.starting_lr = starting_lr
        self.continue_training = continue_training
        self.lognormal = lognormal

        # Instantiate Lipschitz network object, and transfer to device
        self.lipschitz_f = LipschitzFunction(dim=lipschitz_f_dim)
        self.lipschitz_f = self.lipschitz_f.to(self.device)

        # Optimise wrt Lipschitz network parameters (theta), Tran used Adagrad optimiser
        self.optimiser = torch.optim.Adagrad(self.lipschitz_f.parameters(), lr=wasserstein_lr)

    def calculate(self, nnet_samples, gp_samples):
        """
        Function for computing the unregularised Lipschitz loss, and the Wasserstein-1 distance estimate.

        :param nnet_samples: torch.Tensor, samples from BNN prior
        :param gp_samples: torch.Tensor, samples from GP prior
        :return: torch.Tensor, contains Wasserstein distance (one entry)
        """
        f_samples = self.lipschitz_f(nnet_samples.T)
        f_gp = self.lipschitz_f(gp_samples.T)
        return torch.mean(torch.mean(f_samples, 0) - torch.mean(f_gp, 0))

    def compute_gradient_penalty(self, samples_p, samples_q):
        """
        Function computing gradient penalty in Lipschitz loss.

        :param samples_p: torch.Tensor, samples from first prior
        :param samples_q: torch.Tensor, samples from second prior
        :return: tuple, gradient penalty without penalty coefficient, average gradient L2 norm
        """
        eps = torch.rand(samples_p.shape[1], 1).to(samples_p.device)  # standard uniform vector
        X = eps * samples_p.t().detach() + (1 - eps) * samples_q.t().detach()  # compute f_hat (or X) for all samples
        X.requires_grad = True  # store gradients for autodiff
        Y = self.lipschitz_f(X)  # Lipschitz function evaluated at f_hat (or X)

        # Compute and return sum of gradients of Lipschitz outputs wrt inputs (applying chain rule)
        gradients = torch.autograd.grad(Y, X,
                                        grad_outputs=torch.ones(Y.size(), device=self.device),
                                        create_graph=True,
                                        retain_graph=True)[0]
        f_gradient_norm = gradients.norm(2, dim=1)  # L2 norm of gradient

        # Output penalty term without penalty coefficient, along with average gradient norm
        grad_penalty, avg_grad_norm = ((f_gradient_norm - 1) ** 2).mean(), f_gradient_norm.mean().item()
        return grad_penalty, avg_grad_norm

    def lipschitz_optimisation(self, X, n_samples, out_dir, n_steps=200, print_every=10, outer_step=None):
        """
        Performs inner Lipschitz optimisation loop.

        :param X: torch.Tensor, size (n_data, 1), measurement set
        :param n_samples: int, number of BNN and GP samples (N_s in paper)
        :param n_steps: int, number of loop repeats (n_Lipschitz in paper)
        :param print_every: int, regularity of printed feedback in outer optimisation loop
        :param outer_step: int, current step in outer optimisation loop
        """
        # Enable storing gradients for parameters
        for p in self.lipschitz_f.parameters():
            p.requires_grad = True

        # Transfer measurement set to CPU if gpu_gp specified as false
        if not self.gpu_gp:
            X = X.to("cpu")

        # Draw functions from GP, into tensor of size (n_data, n_samples)
        if self.raw_data:
            gp_samples_bag = self.gp.sample_functions(n_samples, lognormal = self.lognormal).detach().float().to(self.device)
        else:
            gp_samples_bag = self.gp.sample_functions(X.double(), n_samples, lognormal = self.lognormal).detach().float().to(self.device)

        # Retrieve measurement set if transferred to CPU earlier
        if not self.gpu_gp:
            X = X.to(self.device)

        # Draw functions from BNN, into tensor of size (n_data, n_samples)
        if self.save_memory:
            nnet_samples_bag = torch.empty((X.shape[0], n_samples)).detach().float().to(self.device)
            nnet_samples_bag[:, :n_samples // 4] \
                = self.bnn.sample_functions(X, n_samples // 4).detach().float().squeeze().to(self.device)
            nnet_samples_bag[:, n_samples // 4:n_samples // 2] \
                = self.bnn.sample_functions(X, n_samples // 4).detach().float().squeeze().to(self.device)
            nnet_samples_bag[:, n_samples // 2:3 * n_samples // 4] \
                = self.bnn.sample_functions(X, n_samples // 4).detach().float().squeeze().to(self.device)
            nnet_samples_bag[:, 3 * n_samples // 4:] \
                = self.bnn.sample_functions(X, n_samples // 4).detach().float().squeeze().to(self.device)
        else:
            nnet_samples_bag = self.bnn.sample_functions(X, n_samples).detach().float().squeeze().to(self.device)

        ###########################################
        # Turn GP and BNN samples into an iterable

        # Resize samples for use in DataLoader (need first index to identify each sample)
        gp_samples_bag = gp_samples_bag.transpose(0, 1)  # resize from (n_data, n_samples) to (n_samples, n_data)
        nnet_samples_bag = nnet_samples_bag.transpose(0, 1)  # swap first two modes of (n_data, n_samples, n_out)

        # Dataset wrapping tensors: each sample retrieved by indexing tensors along the first index
        dataset = TensorDataset(gp_samples_bag, nnet_samples_bag)

        # Combines a dataset and a sampler, providing an iterable over the given dataset
        # batch size: how many samples per batch to load (batches not shuffled)
        # num_workers: how many subprocesses to use for data loading; 0 means data will be loaded in the main process
        data_loader = DataLoader(dataset, batch_size=n_samples, num_workers=0)

        # Takes input DataLoader iterable, and outputs an iterator object which can be cycled through indefinitely
        batch_generator = itertools.cycle(data_loader)

        ###########################################

        # Initialise lists to contain NN gradient norms and parameter gradient norms
        f_grad_norms = []
        p_grad_norms = []
        lip_losses = []

        for i in range(n_steps):
            gp_samples, nnet_samples = next(batch_generator)

            # Resize samples again, for use with our own functions
            gp_samples = gp_samples.transpose(0, 1)  # resize from (n_samples, n_data) to (n_data, n_samples)
            nnet_samples = nnet_samples.transpose(0, 1)  # swap first two modes of (n_samples, n_data, n_out)

            # Create Lipschitz loss objective, augmented with penalty
            objective = -self.calculate(nnet_samples, gp_samples)  # negative unregularised loss
            penalty = self.compute_gradient_penalty(nnet_samples, gp_samples)[0]
            objective += self.penalty_coeff * penalty  # add gradient penalty

            # Note: minimise (-loss + penalty) to maximise (loss - penalty), which estimates the Wasserstein distance

            # Store augmented Lipschitz loss containing values for all samples
            lip_losses.append(-1 * objective.item())  # important: use negative

            # Minimise above objective wrt Lipschitz network parameters
            self.optimiser.zero_grad()  # set all gradients to zero (prevent accumulation)
            objective.backward()  # populate p.grad with dL/dp for each network parameter p, scalar objective L
            # torch.nn.utils.clip_grad_norm_(self.lipschitz_f.parameters(), 5000.)  # prevent exploding gradients

            if self.continue_training:
                if outer_step == 1:
                    self.optimiser.param_groups[0]["lr"] = self.starting_lr
                elif outer_step == 2:
                    self.optimiser.param_groups[0]["lr"] = self.wasserstein_lr
            self.optimiser.step()  # perform one minimisation step using gradients from backward() call

            # Compute norm of NN gradient in the penalty term; store in list containing values for all samples
            f_grad_norms.append(self.compute_gradient_penalty(nnet_samples, gp_samples)[1])

            # Compute norm of parameter gradients dL/dp to assess convergence of the network parameters
            params = self.lipschitz_f.parameters()  # obtain Lipschitz network parameters
            grad_norm = torch.cat([p.grad.detach().flatten() for p in params]).norm()

            # The value of grad_norm above is computed as follows:
            # 1. Fill a tensor with the Lipschitz network gradients dL/dp (loss L and parameter p)
            # 2. Compute the Frobenius norm of this tensor
            # Shrinking grad_norm indicates convergence of network parameters (smaller steps in Adagrad optimiser)

            # Store parameter gradient norm in list containing values for all samples
            p_grad_norms.append(grad_norm.item())

            if (outer_step % print_every == 0 or outer_step == 1) & (((i+1) % 50 == 0) or (i in [0, 9, 19, 29, 39])):
                print('Grad norm %.3f at step %d' % (grad_norm, i+1))

        # Reset gradient requirement, since optimisation is complete
        for p in self.lipschitz_f.parameters():
            p.requires_grad = False

        ensure_dir(out_dir)
        if outer_step > 1:
            # Save penalty term NN gradient norms in a binary file
            nn_file = os.path.join(out_dir, "f_grad_norms")  # saved as .npy file
            f_grad_norms = np.array(f_grad_norms)
            np.save(nn_file, f_grad_norms)

            # Do the same for concatenated parameter gradient norms
            param_file = os.path.join(out_dir, "p_grad_norms")  # saved as .npy file
            p_grad_norms = np.array(p_grad_norms)
            np.save(param_file, p_grad_norms)

            # Do the same for Lipschitz losses
            loss_file = os.path.join(out_dir, "lip_losses")  # saved as .npy file
            lip_losses = np.array(lip_losses)
            np.save(loss_file, lip_losses)

class MapperWasserstein(object):
    def __init__(self, gp, bnn, data_generator, out_dir, n_data=256, wasserstein_steps=200, wasserstein_lr=0.02,
                 starting_steps=1000, starting_lr=0.001, n_gpu=0, gpu_gp=False, save_memory=False, raw_data=False,
                 continue_training=True, lognormal=False):
        """
        Implementation of Wasserstein distance minimisation.

        :param gp: nn.Module (instance of Kern child), GP prior
        :param bnn: nn.Module, BNN prior
        :param data_generator: instance of data generation object (e.g. GridGenerator), generates measurement set
        :param out_dir: str, specify directory for output files (containing checkpoint data)
        :param n_data: int, size of measurement set
        :param wasserstein_steps: int, number of inner Lipschitz loop repeats
        :param wasserstein_lr: float, learning rate for Lipschitz optimiser
        :param starting_steps: int, number of inner steps at first outer iteration (optional)
        :param starting_lr: float, learning rate for Lipschitz optimiser at first outer iteration (optional)
        :param n_gpu: int, number of GPUs to utilise
        :param gpu_gp: bool, specify whether to compute GP samples on GPU or not
        :param save_memory: bool, specify if memory demands should be reduced at expense of slower computation
        :param raw_data: bool, specify if raw data samples are being used as the target prior
        :param continue_training: bool, specify if the Lipschitz network is pretrained or not
        :param lognormal: bool, specify if process is lognormal or not (exponentiate GP samples or not)
        """
        self.gp = gp
        self.bnn = bnn
        self.data_generator = data_generator
        self.n_data = n_data
        self.out_dir = out_dir
        self.device, device_ids = prepare_device(n_gpu)
        self.gpu_gp = gpu_gp
        self.wasserstein_steps = wasserstein_steps
        self.starting_steps = starting_steps
        self.save_memory = save_memory
        self.raw_data = raw_data
        self.continue_training = continue_training
        self.lognormal = lognormal

        # Move models to configured device
        if gpu_gp:
            self.gp = self.gp.to(self.device)
        self.bnn = self.bnn.to(self.device)

        # Parallelisation support for multiple devices
        if len(device_ids) > 1:
            if self.gpu_gp:
                self.gp = torch.nn.DataParallel(self.gp, device_ids=device_ids)
            self.bnn = torch.nn.DataParallel(self.bnn, device_ids=device_ids)

        # Initialise the WassersteinDistance module
        self.wasserstein = WassersteinDistance(bnn, gp,
                                               lipschitz_f_dim=n_data,
                                               wasserstein_lr=wasserstein_lr,
                                               starting_lr=starting_lr,
                                               device=self.device,
                                               gpu_gp=gpu_gp,
                                               save_memory=save_memory,
                                               raw_data=raw_data,
                                               continue_training=continue_training,
                                               lognormal=lognormal)

        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)

    def optimise(self, num_iters, n_samples=128, lr=0.05, print_every=10, save_ckpt_every=50):
        """
        Implement outer optimisation loop for BNN prior hyperparameters.

        :param num_iters: int, number of outer loop repeats
        :param n_samples: int, number of GP and BNN samples (N_s in paper)
        :param lr: float, learning rate of outer optimiser
        :param print_every: int, frequency of printed feedback
        :param save_ckpt_every: int, frequency of save checkpoints
        :return: list, Wasserstein distance history (for plotting)
        """
        wdist_hist = []

        # Optimise wrt BNN prior hyperparameters; Tran used RMSprop optimiser
        prior_optimizer = torch.optim.RMSprop(self.bnn.parameters(), lr=lr)

        # Note: for each layer have W_std = softplus(W_rho) and b_std = softplus(b_rho), with rho values optimised.
        #       The rho values are registered as parameters using nn.Parameter, with the setting require_grad = True.

        # Outer optimisation loop for BNN prior hyperparameters
        for it in range(1, num_iters+1):

            # Generate measurement set
            X = self.data_generator.get(self.n_data)  # size (n_data, input_dim)
            X = X.to(self.device)

            # Transfer measurement set to CPU if gpu_gp specified as false
            if not self.gpu_gp:
                X = X.to("cpu")

            # Draw functions from GP
            if self.raw_data:
                if self.n_data != 64**2:
                    raise Exception('Only 64-by-64 SST samples are considered; need n_data = 64^2')
                gp_samples = self.gp.sample_functions(n_samples, lognormal = self.lognormal).detach().float().to(self.device)
            else:
                gp_samples = self.gp.sample_functions(X.double(), n_samples, lognormal = self.lognormal).detach().float().to(self.device)

            # Retrieve measurement from CPU if transferred earlier
            if not self.gpu_gp:
                X = X.to(self.device)

            # Draw functions from BNN
            if self.save_memory:
                nnet_samples = torch.empty((X.shape[0], n_samples)).float().to(self.device)
                nnet_samples[:, :n_samples // 4] \
                    = self.bnn.sample_functions(X, n_samples // 4).float().squeeze().to(self.device)
                nnet_samples[:, n_samples // 4:n_samples // 2] \
                    = self.bnn.sample_functions(X, n_samples // 4).float().squeeze().to(self.device)
                nnet_samples[:, n_samples // 2:3 * n_samples // 4] \
                    = self.bnn.sample_functions(X, n_samples // 4).float().squeeze().to(self.device)
                nnet_samples[:, 3 * n_samples // 4:] \
                    = self.bnn.sample_functions(X, n_samples // 4).float().squeeze().to(self.device)
            else:
                nnet_samples = self.bnn.sample_functions(X, n_samples).float().squeeze().to(self.device)

            # Initialise parameters of Lipschitz neural net
            if not self.continue_training or it == 1:
                self.wasserstein.lipschitz_f.reset_parameters()
            if self.continue_training and it == 1:
                inner_steps = self.starting_steps
            else:
                inner_steps = self.wasserstein_steps

            # Inner optimisation loop to maximise augmented Lipschitz loss wrt network parameters (theta)
            self.wasserstein.lipschitz_optimisation(X, n_samples,
                                                    out_dir=self.out_dir,
                                                    n_steps=inner_steps,
                                                    print_every=print_every,
                                                    outer_step=it)

            # Load penalty term gradients for this particular outer optimisation step
            ensure_dir(self.out_dir)
            nn_file = os.path.join(self.out_dir, "f_grad_norms.npy")  # file is overwritten at each step
            param_file = os.path.join(self.out_dir, "p_grad_norms.npy")  # ditto
            loss_file = os.path.join(self.out_dir, "lip_losses.npy")  # ditto

            # Store penalty term gradients for all outer optimisation steps (one row per outer step, before transpose)
            if it == 2:
                f_grad_norms = np.load(nn_file)
                p_grad_norms = np.load(param_file)
                lip_losses = np.load(loss_file)
            elif it > 2:
                f_grad_norms = np.vstack((f_grad_norms, np.load(nn_file)))
                p_grad_norms = np.vstack((p_grad_norms, np.load(param_file)))
                lip_losses = np.vstack((lip_losses, np.load(loss_file)))

            # Minimise Wasserstein distance (after Lipschitz maximisation) wrt BNN hyperparameters (psi)
            prior_optimizer.zero_grad()  # set gradients to zero (prevent accumulation)
            wdist = self.wasserstein.calculate(nnet_samples, gp_samples)  # compute Wasserstein distance
            wdist.backward()  # compute gradients of Wasserstein distance wrt hyperparameters
            prior_optimizer.step()  # update BNN rho values using gradients from backward() call

            # Store Wasserstein distance and print iteration info
            wdist_hist.append(float(wdist))
            if (it % print_every == 0) or it == 1:
                print(">>> Iteration # {:3d}: Wasserstein Dist {:.4f}".format(it, float(wdist)))

            # Save checkpoint
            if (it % save_ckpt_every == 0) or (it in [1, 10, num_iters]):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                torch.save(self.bnn.state_dict(), path)

        # Store the gradient norms and losses for all outer optimisation steps
        ensure_dir(self.out_dir)
        nn_file = os.path.join(self.out_dir, "f_grad_norms")  # saved as .npy file
        param_file = os.path.join(self.out_dir, "p_grad_norms")  # ditto
        loss_file = os.path.join(self.out_dir, "lip_losses")  # ditto
        np.save(nn_file, f_grad_norms.T)  # one column per outer step, after transpose
        np.save(param_file, p_grad_norms.T)  # ditto
        np.save(loss_file, lip_losses.T)  # ditto

        # Return history of Wasserstein distance values (for assessing convergence)
        return wdist_hist
