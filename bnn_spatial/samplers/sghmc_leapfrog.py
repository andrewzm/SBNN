"""
Sampler for SGHMC, with leapfrog discretisation
"""

import torch
import numpy as np
from torch.optim import Optimizer


class SGHMC(Optimizer):
    def __init__(self, params, lr=1e-4, mdecay=0.05, scale_grad=1.):
        """
        SGHMC sampler with one leapfrog step.

        References:
        [1] https://arxiv.org/pdf/1402.4102.pdf

        Note: In [1] eq. 15, we set beta_hat=0 since we estimate B_hat=0, assuming lr is suitably small

        :param params: iterable, parameters serving as optimization variable
        :param lr: float, base learning rate for this optimizer (eta in [1] eq. 15)
        :param mdecay: float, momentum decay per time-step (alpha in [1] eq. 15)
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

        for param_idx, parameter in enumerate(group["params"]):

            if parameter.grad is None:
                continue

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
            sigma = torch.sqrt(torch.from_numpy(np.array(lr * mdecay / 2, dtype=type(lr))))
            sample_t = torch.normal(mean=torch.zeros_like(gradient),
                                    std=torch.ones_like(gradient) * sigma)

            # Update momentum by one half-step
            momentum.add_(-lr/2 * gradient - mdecay/2 * momentum + sample_t)

            # Update parameter
            parameter.detach().add_(momentum)

            # Update momentum by another half-step
            gradient = parameter.grad.detach() * scale_grad
            momentum.add_(-lr/2 * gradient - mdecay/2 * momentum + sample_t)

