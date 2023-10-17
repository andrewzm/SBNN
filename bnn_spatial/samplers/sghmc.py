"""
Sampler for SGHMC, with modified Euler discretisation
"""

import torch
import numpy as np
from torch.optim import Optimizer

class SGHMC(Optimizer):
    def __init__(self, params, lr=1e-4, mdecay=0.05, scale_grad=1.):
        """
        Stochastic gradient Hamiltonian Monte Carlo sampler.

        References:
        [1] https://arxiv.org/pdf/1402.4102.pdf

        :param params: iterable, parameters serving as optimization variable
        :param lr: float, base learning rate for this optimizer
        :param mdecay: float, momentum decay per time-step
        :param scale_grad: float, scaling factor for mini-batch gradient, number of examples in the entire dataset
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr,
            scale_grad=scale_grad,
            mdecay=mdecay
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

        # For parameter tensor dimensions, see adaptive_sghmc.py comments

        for param_idx, parameter in enumerate(group["params"]):

            if parameter.grad is None:
                raise ValueError('Parameter gradient is None')
            if torch.any(torch.isnan(parameter.grad)):
                raise ValueError('NaN values in parameter gradient')

            state = self.state[parameter]

            if len(state) == 0:
                state["iteration"] = 0
                state["momentum"] = torch.zeros_like(parameter)

            state["iteration"] += 1

            mdecay, lr = group["mdecay"], group["lr"]
            scale_grad = torch.tensor(group["scale_grad"], dtype=parameter.dtype)

            momentum = state["momentum"]
            gradient = parameter.grad.detach() * scale_grad

            # Sample random epsilon
            sigma = torch.sqrt(torch.from_numpy(np.array(2 * lr * mdecay, dtype=type(lr))))
            sample_t = torch.normal(mean=torch.zeros_like(gradient),
                                    std=torch.ones_like(gradient) * sigma)

            # Update momentum (Eq 15 below in [1])
            momentum.add_(-lr * gradient - mdecay * momentum + sample_t)

            # Update parameter (Eq 15 above in [1])
            parameter.detach().add_(momentum)
