import copy
import pathlib
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn, optim


class CheckpointManager(object):
    r"""
    A :class:`CheckpointManager` periodically serializes models and optimizer
    as .pth files during training, and optionally keeps track of best performing
    checkpoint based on an observed metric. The filenames of serialized
    checkpoints are ``model_{iteration}.pth`` and ``optimizer_{iteration}.pth``.

    Note
    ----
    For :class:`~torch.nn.parallel.DistributedDataParallel` objects,
    ``.module.state_dict()`` is called instead of ``.state_dict()`` in
    :meth:`step`.

    Parameters
    ----------
    model: torch.nn.Module
        Model which needs to be serialized as a checkpoint.
    optimizer: torch.optim.Optimizer
        Optimizer which needs to be serialized as a checkpoint.
    serialization_dir: str
        Path to an empty or non-existent directory to save checkpoints.
    mode: str, optional (default=None)
        One of ``{"min", "max", None}``. In ``min`` mode, best checkpoint will
        be recorded when metric hits a lower value; in ``max`` mode it will be
        recorded when metric hits a higher value, else best checkpoint won't
        be kept track of when ``None``.
    k_recent: int, optional (default=None)
        Number of recent 'k' checkpoints to keep on disk. Older checkpoints
        will be removed. Set to ``-1`` for keeping all checkpoints.

    Examples
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.SGD(model.parameters())
    >>> ckpt_manager = CheckpointManager(model, optimizer, "/tmp", mode="min")
    >>> num_epochs = 20
    >>> for epoch in range(num_epochs):
    ...     train(model)
    ...     val_loss = validate(model)
    ...     ckpt_manager.step(epoch, metric=val_loss)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        serialization_dir: str,
        mode: Optional[str] = None,
        k_recent: int = 100,
    ):
        self._model = model
        self._optimizer = optimizer
        self._serialization_dir = pathlib.Path(serialization_dir)
        self._mode = mode
        self._k_recent = k_recent

        # Initialize members to hold state dict of best checkpoint and its
        # performance.
        self._best_metric: Optional[Union[float, torch.Tensor]] = None
        self._best_ckpt: Dict[str, Any] = {}

        # Keep epoch/iteration numbers of recently saved 'n' checkpoints.
        self._recent_iterations: List[int] = []

    def step(self, epoch_or_iteration: int, metric: Optional[float] = None):
        r"""
        Serialize checkpoint and update best checkpoint based on metric and
        mode.
        """
        if isinstance(self._model, nn.parallel.DistributedDataParallel):
            model_state_dict = self._model.module.state_dict()
        else:
            model_state_dict = self._model.state_dict()

        # Serialize checkpoint corresponding to current epoch (or iteration).
        torch.save(
            model_state_dict,
            self._serialization_dir / f"model_{epoch_or_iteration}.pth",
        )
        torch.save(
            self._optimizer.state_dict(),
            self._serialization_dir / f"optimizer_{epoch_or_iteration}.pth",
        )
        self._recent_iterations.append(epoch_or_iteration)

        # Remove earliest checkpoint if there are more on disk.
        if self._k_recent > 0 and len(self._recent_iterations) > self._k_recent:
            self.remove_earliest_checkpoint()

        # Update best checkpoint based on metric and metric mode.
        if not self._best_metric and self._mode is not None:
            self._best_metric = metric

        if (self._mode == "min" and metric < self._best_metric) or (
            self._mode == "max" and metric > self._best_metric
        ):
            self._best_metric = metric
            self._best_ckpt = copy.copy(model_state_dict)

        # Serialize best performing checkpoint observed so far.
        if self._mode is not None:
            torch.save(self._best_ckpt, self._serialization_dir / f"model_best.pth")

    def remove_earliest_checkpoint(self):
        earliest_iteration = self._recent_iterations.pop(0)
        (self._serialization_dir / f"model_{earliest_iteration}.pth").unlink()
        (self._serialization_dir / f"optimizer_{earliest_iteration}.pth").unlink()
