"""
Sampler for scale-adapted (adaptive) SGHMC, with modified Euler discretisation
"""

import torch
import numpy as np

from torch.optim import Optimizer

# 09/10: Replace .data with .detach(): https://stackoverflow.com/questions/51743214/is-data-still-useful-in-pytorch

class AdaptiveSGHMC(Optimizer):
    def __init__(self, params, lr=1e-2, num_burn_in_steps=3000, epsilon=1e-8, mdecay=0.05, scale_grad=1.):
        """
        SGHMC sampler that automatically adapts its own hyperparameters during the burn-in phase

        References:
        [1] http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf
        [2] https://arxiv.org/pdf/1402.4102.pdf

        :param params: iterable, parameters serving as optimization variable
        :param lr: float, base learning rate for this optimizer
        :param num_burn_in_steps: int, number of burn-in steps to perform (in each burn-in step, this sampler will adapt
            its own internal parameters to decrease its error; set to `0` to turn scale adaption off)
        :param epsilon: float, per-parameter epsilon level
        :param mdecay: float, momentum decay per time-step
        :param scale_grad: float, scaling factor for mini-batch gradient, number of examples in the entire dataset
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr,
            scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            epsilon=epsilon
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform one optimisation step on each network parameter.

        :param closure: an optional callable enabling reevaluation of loss at multiple time steps (not used)
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Parameter group is a dict of parameter:value pairs (there is only one parameter group in this case)
        group = self.param_groups[0]

        # Iterate over tensors containing weights/biases for each layer, initially given by net.parameters()
        for param_idx, parameter in enumerate(group["params"]):

            if parameter.grad is None:
                raise ValueError('Parameter gradient is None')
            if torch.any(torch.isnan(parameter.grad)):
                raise ValueError('NaN values in parameter gradient')

            # Access current parameter state, where state is a dict with current configuration of all parameters
            state = self.state[parameter]

            # If this is the first time accessing the state of a given parameter, set the following defaults
            if len(state) == 0:
                state["iteration"] = 0
                state["tau"] = torch.ones_like(parameter)
                state["g"] = torch.ones_like(parameter)
                state["v_hat"] = torch.ones_like(parameter)
                state["momentum"] = torch.zeros_like(parameter)
            state["iteration"] += 1

            # Gather parameters from group and state dicts to be used in the update computation
            mdecay = group["mdecay"]
            epsilon = group["epsilon"]
            lr = group["lr"]
            scale_grad = torch.tensor(group["scale_grad"], dtype=parameter.dtype)  # training set size

            tau, g, v_hat = state["tau"], state["g"], state["v_hat"]  # for burn-in adaptation phase

            momentum = state["momentum"]
            gradient = parameter.grad.detach() * scale_grad
            tau_inv = 1. / (tau + 1.)

            # Update parameters during burn-in
            if state["iteration"] <= group["num_burn_in_steps"]:
                # Specifies the moving average window, see Eq 9 in [1] left
                tau.add_(- tau * (g**2 / (v_hat + epsilon)) + 1)  # edited, had g*g before

                # Average gradient see Eq 9 in [1] right
                g.add_(-g * tau_inv + tau_inv * gradient)

                # Gradient variance see Eq 8 in [1]
                v_hat.add_(-v_hat * tau_inv + tau_inv * (gradient ** 2))

            # Note: add_ is the in-place version of add, which directly alters tensor without making a copy
            # e.g. if x=y=[1], then x.add(y)=[2] with x=[1], whereas x.add_(y)=[2] with x=[2]

            # Preconditioner
            minv_t = 1. / (torch.sqrt(v_hat) + epsilon)

            # Variance of noise see Eq 10 in [1] right, the Gaussian term
            epsilon_var = (2. * (lr ** 2) * mdecay * minv_t - (lr ** 4))

            # Sample Gaussian noise (for the Gaussian term)
            sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))
            sample_t = torch.normal(mean=torch.zeros_like(gradient),
                                    std=torch.ones_like(gradient) * sigma)

            # Update momentum (Eq 10 right in [1])
            momentum.add_(
                - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
            )

            # Update parameters (Eq 10 left in [1])
            parameter.detach().add_(momentum)
