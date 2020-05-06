import bisect
import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupNoDecayLR(LambdaLR):
    r"""
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further keeps it constant throughout training.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper optimizer.
    total_steps: int
        Total epochs (or iterations) for training.
    warmup_steps: int
        Number of first few steps to do linear warmup.
    last_epoch: int, optional (default = -1)
        The index of last step (epoch or iteration). We named it ``last_epoch``
        instead of ``last_step`` to keep the naming consistent with other LR
        schedulers in PyTorch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        assert (
            warmup_steps < total_steps
        ), "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        multiplier = step / float(max(1, self.wsteps)) if step < self.wsteps else 1
        return max(0, multiplier)


class LinearWarmupMultiStepLR(LambdaLR):
    r"""
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it by gamma once the number of steps reaches one
    of the milestones.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper optimizer.
    total_steps: int
        Total epochs (or iterations) for training.
    warmup_steps: int
        Number of first few steps to do linear warmup.
    milestones: List[int]
        List of step indices (epochs or iterations depending on context). Must
        be increasing.
    gamma: float, optional (default = 0.1)
        Multiplicative factor of learning rate decay.
    last_epoch: int, optional (default = -1)
        The index of last step (epoch or iteration). We named it ``last_epoch``
        instead of ``last_step`` to keep the naming consistent with other LR
        schedulers in PyTorch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.wsteps = warmup_steps
        self.milestones = milestones
        self.gamma = gamma

        # Keep a track of number of milestones encountered.
        self.milestones_so_far = 0

        # Common sanity checks.
        assert milestones == sorted(milestones), "milestones must be increasing"
        assert milestones[0] > warmup_steps, "first milestone must be after warmup"
        assert (
            milestones[-1] < total_steps
        ), "last milestone must be less than total steps"

        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Step decay based on milestones.
            multiplier = self.gamma ** bisect.bisect_right(self.milestones, step)

        # Avoid negative learning rate.
        return max(0, multiplier)


class LinearWarmupLinearDecayLR(LambdaLR):
    r"""
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it linearly to zero.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper optimizer.
    total_steps: int
        Total epochs (or iterations) for training.
    warmup_steps: int
        Number of first few steps to do linear warmup.
    last_epoch: int, optional (default = -1)
        The index of last step (epoch or iteration). We named it ``last_epoch``
        instead of ``last_step`` to keep the naming consistent with other LR
        schedulers in PyTorch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        assert (
            warmup_steps < total_steps
        ), "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Linear decay.
            multiplier = (self.tsteps - step) / (self.tsteps - self.wsteps)
        # Avoid negative learning rate.
        return max(0, multiplier)


class LinearWarmupCosineAnnealingLR(LambdaLR):
    r"""
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it to zero by cosine decay. After linear warmup,
    the LR decays as:

    .. math::
        \eta_t = \eta_{max}\cos^2(\frac{T_{cur} - T_{warm}}{T_{max} - T_{warm}}\frac{\pi}{2})

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper optimizer.
    total_steps: int
        Total epochs (or iterations) for training.
    warmup_steps: int
        Number of first few steps to do linear warmup.
    last_epoch: int, optional (default = -1)
        The index of last step (epoch or iteration). We named it ``last_epoch``
        instead of ``last_step`` to keep the naming consistent with other LR
        schedulers in PyTorch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        assert (
            warmup_steps < total_steps
        ), "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Cosine annealing decay.
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        # Avoid negative learning rate.
        return max(0, multiplier)
