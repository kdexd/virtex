from typing import Optional, Type

from torch import optim
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLinearDecayLR(LambdaLR):
    r"""
    A learning rate scheduler which increases learning rate in a linearly from
    0 to maximum (initial) lr, and decreases it linearly to zero.

    .. note::

        This class closely follows the API of other PyTorch LR schedulers.
        We use "epoch" in variable names for consistency, however it may refer
        to iterations. It is just a counter for the number of :meth:`step` calls
        of this scheduler.

    Parameters
    ----------
    optimizer: Type[torch.optim.Optimizer]
        Wrapper optimizer.
    total_epochs: int
        Total epochs (or iterations) for training.
    warmup_proportion: float
        A value in ``[0, 1)``. The warmup happens for first ``total_epochs *
        warmup_proportion`` steps. Linear decay after that until the end.
    last_epoch: int, optional (default = -1)
        The index of last epoch.
    """

    def __init__(
        self,
        optimizer: Type[optim.Optimizer],
        total_epochs: int,
        warmup_proportion: float,
        last_epoch: Optional[int] = -1,
    ):
        assert warmup_proportion < 1, "Warmup proportion should be in [0, 1)."
        self._total_epochs = total_epochs
        self._warmup_epochs = int(total_epochs * warmup_proportion)
        super().__init__(optimizer, self._lr_lambda, last_epoch)

    def _lr_lambda(self, epoch: int) -> float:
        r"""
        Lambda function for the super class. This returns a multiplier (a value
        in ``[0, 1]`` for the optimizer's lr, depending on the current step.

        Parameters
        ----------
        epoch: int,
            Current epoch (o iteration). In other words, a counter for number
            of :meth:`step` calls. This function is used by the super class.

        Returns
        -------
        float
            A multiplier factor for the optimizer's lr.
        """
        if epoch < self._warmup_epochs:
            # Linear warmup.
            multiplier = float(epoch) / float(max(1, self._warmup_epochs))
        else:
            # Linear decay.
            multiplier = float(self._total_epochs - epoch) / float(
                max(1, self._total_epochs - self._warmup_epochs)
            )
        # Avoid negative learning rate.
        return max(0, multiplier)
