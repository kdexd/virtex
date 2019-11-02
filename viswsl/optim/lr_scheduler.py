from typing import Type

from torch import optim
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLinearDecayLR(LambdaLR):
    r"""
    A learning rate scheduler which increases learning rate in a linearly from
    0 to maximum (initial) lr, and decreases it linearly to zero.

    Parameters
    ----------
    optimizer: Type[torch.optim.Optimizer]
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
        optimizer: Type[optim.Optimizer],
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        assert warmup_steps < total_steps, "Warmup steps should be less than total steps."
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps
        super().__init__(optimizer, self._lr_lambda, last_epoch)

    def _lr_lambda(self, step: int) -> float:
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
        if step < self._warmup_steps:
            # Linear warmup.
            multiplier = float(step) / float(max(1, self._warmup_steps))
        else:
            # Linear decay.
            multiplier = float(self._total_steps - step) / float(
                max(1, self._total_steps - self._warmup_steps)
            )
        # Avoid negative learning rate.
        return max(0, multiplier)
