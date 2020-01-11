import math

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
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step: int) -> float:
        r"""
        Lambda function for the super class. This returns a multiplier (a value
        in ``[0, 1]`` for the optimizer's lr, depending on the current step.

        Parameters
        ----------
        step: int,
            Current step (epoch or iteration). Used by the super class.

        Returns
        -------
        float
            A multiplier factor for the optimizer's lr.
        """
        multiplier = step / float(max(1, self.wsteps)) if step < self.wsteps else 1
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
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step: int) -> float:
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
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Cosine annealing decay.
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        # Avoid negative learning rate.
        return max(0, multiplier)
