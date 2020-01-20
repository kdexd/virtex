import os
from typing import Dict, Union

from loguru import logger
import torch
from torch import distributed as dist


def init_distributed_env(backend: str = "nccl") -> int:
    r"""
    Initialize distributed process group from five environment variables:
    ``$MASTER_ADDR, $MASTER_PORT, $WORLD_SIZE, $RANK, $LOCAL_RANK``. Suitable
    and recommended if you are using SLURM.

    ``$LOCAL_RANK`` will be equal to ``$RANK`` for single machine multi-GPU
    training. If we are using multi-node multi-GPU, for example: two machines
    with 2 GPUs each. The process group woud have four processes with ``$RANK``s
    (0, 1, 2, 3) and ``$LOCAL_RANK``s (0, 1, 0, 1).

    Note
    ----
    If you are using SLURM, you only need to set ``$MASTER_PORT`` -- this method
    would take the rest from env variables set by SLURM.

    Note
    ----
    Use NCCL Backend for training, GLOO backend for debugging.

    Parameters
    ----------
    backend: str, optional (default = "nccl")
        Backend for :mod:`torch.distributed`, either "gloo" or "nccl".

    Returns
    -------
    int
        Device ID of the GPU used by current process.
    """
    assert torch.cuda.is_available(), "Cannot use GPU, CUDA not found!"

    # Set env variables required to initialize distributed process group.
    # If using SLURM, these may have been set as some other name.
    os.environ["MASTER_ADDR"] = os.environ.get(
        "MASTER_ADDR", os.environ.get("SLURM_NODELIST", "localhost").split(",")[-1]
    )
    os.environ["RANK"] = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ["WORLD_SIZE"] = os.environ.get(
        "WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")
    )
    try:
        dist.init_process_group(backend, init_method="env://")
        # Wait for all processes to initialize, necessary to avoid timeout.
        synchronize()
    except Exception as e:
        logger.error(
            f"Dist URL: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        raise e

    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    )
    # Current process only accesses this single GPU exclusive of other processes
    # in the process group.
    torch.cuda.set_device(local_rank)
    return local_rank


def synchronize() -> None:
    r"""Synchronize (barrier) processes in a process group."""
    if dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    r"""Return number of processes in the process group, each uses 1 GPU."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    r"""Return rank of current process in the process group."""
    return dist.get_rank() if dist.is_initialized() else 0


def is_master_process() -> bool:
    r"""
    Check if current process is the master process in distributed training
    process group. useful to make checks while tensorboard logging and
    serializing checkpoints. Always ``True`` for single-GPU single-machine.
    """
    return get_rank() == 0


def average_across_processes(t: Union[Dict[str, torch.Tensor], torch.Tensor]):
    r"""
    Averages a tensor, or a (flat) dict of tensors across all processes in a
    process group. All processes finally have the same mean value.
    """
    if dist.is_initialized():
        if isinstance(t, torch.Tensor):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= get_world_size()
        elif isinstance(t, dict):
            for k in t:
                dist.all_reduce(t[k], op=dist.ReduceOp.SUM)
                t[k] /= dist.get_world_size()
