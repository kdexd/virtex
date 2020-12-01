r"""
`Lookahead Optimizer: k steps forward, 1 step back <https://arxiv.org/abs/1907.08610>`_.

This implementation is adapted with minimal modifications from the
`authors' implementation <https://github.com/michaelrzhang/lookahead>`_.

If you take it from here, please cite them:

.. code-block:: text

    @inproceedings{zhang2019lookahead,
        title={Lookahead Optimizer: k steps forward, 1 step back},
        author={Zhang, Michael R and Lucas, James and Hinton, Geoffrey and Ba, Jimmy},
        journal={NeurIPS},
        year={2019}
    }
"""
from collections import defaultdict
from typing import Any, Callable, Dict

import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    r"""
    Implements Lookahead optimizer.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper inner optimizer. The weights it manages will be the "fast"
        weights.
    k: int, optional (default = 5)
        Number of lookahead steps before updating "slow" weights.
    alpha: float, optional (default = 0.8)
        Linear interpolation factor, 1.0 recovers inner optimizer.
    """

    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.8):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha

        # Counter for inner optimizer.
        self._k_counter = 0

        # Cache the current optimizer parameters
        self.state: Dict[str, Any] = defaultdict(dict)
        for group in optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["slow_params"] = torch.zeros_like(p.data)
                param_state["slow_params"].copy_(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "alpha": self.alpha,
            "k": self.k,
            "_k_counter": self._k_counter,
        }

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        r"""Clear all grad buffers at the start of new forward pass."""
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure: Callable = None):
        r"""
        Perform a single Lookahead optimization step.

        Parameters
        ----------
        closure: Callable, optional (default = None)
            A callable that re-evaluates the model and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._k_counter += 1

        if self._k_counter >= self.k:
            self._k_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(
                        param_state["slow_params"], alpha=1.0 - self.alpha
                    )
                    param_state["slow_params"].copy_(p.data)
        return loss

    def load_slow_weights(self):
        r"""
        Load slow weights from Lookahead optimizer. Useful for performing
        evaluation on the slow weights (which typically generalize better).

        This method backs up fast weights to load them after evaluation. No
        need to call this method if evaluation happens just after a lookahead
        step.
        """
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["slow_params"])

    def restore_fast_weights(self):
        r"""
        Restore fast weights for optimization. Call this after evaluation if
        :meth:`load_slow_weights` was called.
        """
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]
