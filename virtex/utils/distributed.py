r"""
A collection of common utilities for distributed training. These are a bunch of
wrappers over utilities from :mod:`torch.distributed` module, but they do not
raise exceptions in absence of distributed training / CPU-only training, and
fall back to sensible default behavior.
"""
from typing import Callable, Dict, Tuple, Union

from loguru import logger
import torch
from torch import distributed as dist
from torch import multiprocessing as mp


def launch(
    job_fn: Callable,
    num_machines: int = 1,
    num_gpus_per_machine: int = 1,
    machine_rank: int = 0,
    dist_url: str = "tcp://127.0.0.1:23456",
    args=(),
):
    r"""
    Launch a job in a distributed fashion: given ``num_machines`` machines,
    each with ``num_gpus_per_machine`` GPUs, this utility will launch one
    process per GPU. This wrapper uses :func:`torch.multiprocessing.spawn`.

    The user has to launch one job on each machine, manually specifying a
    machine rank (incrementing integers from 0), this utility will adjust
    process ranks per machine. One process on ``machine_rank = 0`` will be
    refered as the *master process*, and the IP + a free port on this machine
    will serve as the distributed process communication URL.

    Default arguments imply one machine with one GPU, and communication URL
    as ``localhost``.

    .. note::

        This utility assumes same number of GPUs per machine with IDs as
        ``(0, 1, 2 ...)``. If you do not wish to use all GPUs on a machine,
        set ``CUDA_VISIBLE_DEVICES`` environment variable (for example,
        ``CUDA_VISIBLE_DEVICES=5,6``, which restricts to GPU 5 and 6 and
        re-assigns their IDs to 0 and 1 in this job scope).

    Parameters
    ----------
    job_fn: Callable
        A callable object to launch. Pass your main function doing training,
        validation etc. here.
    num_machines: int, optional (default = 1)
        Number of machines used, each with ``num_gpus_per_machine`` GPUs.
    num_gpus_per_machine: int, optional (default = 1)
        Number of GPUs per machine, with IDs as ``(0, 1, 2 ...)``.
    machine_rank: int, optional (default = 0)
        A manually specified rank of the machine, serves as a unique identifier
        and useful for assigning global ranks to processes.
    dist_url: str, optional (default = "tcp://127.0.0.1:23456")
        Disributed process communication URL as ``tcp://x.x.x.x:port``. Set
        this as the IP (and a free port) of machine with rank 0.
    args: Tuple
        Arguments to be passed to ``job_fn``.
    """

    assert (
        torch.cuda.is_available()
    ), "CUDA not available, Cannot launch distributed processes."

    world_size = num_machines * num_gpus_per_machine

    # Spawn ``num_gpus_per_machine``` processes per machine, and provide
    # "local process rank" (GPU ID) as the first arg to ``_dist_worker``.
    # fmt: off
    if world_size > 1:
        mp.spawn(
            _job_worker,
            nprocs=num_gpus_per_machine,
            args=(
                job_fn, world_size, num_gpus_per_machine, machine_rank, dist_url, args
            ),
            daemon=False,
        )
    else:
        # Default to single machine, single GPU, with ID 0.
        _job_worker(0, job_fn, 1, 1, 0, dist_url, args)
    # fmt: on


def _job_worker(
    local_rank: int,
    job_fn: Callable,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: Tuple,
):
    r"""
    Single distibuted process worker. This should never be used directly,
    only used by :func:`launch`.
    """

    # Adjust global rank of process based on its machine rank.
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        logger.error(f"Error launching processes, dist URL: {dist_url}")
        raise e

    synchronize()
    # Set GPU ID for each process according to its rank.
    torch.cuda.set_device(local_rank)
    job_fn(*args)


def synchronize() -> None:
    r"""Synchronize (barrier) all processes in a process group."""
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
    Check whether current process is the master process. This check is useful
    to restrict logging and checkpointing to master process. It will always
    return ``True`` for single machine, single GPU execution.
    """
    return get_rank() == 0


def average_across_processes(t: Union[torch.Tensor, Dict[str, torch.Tensor]]):
    r"""
    Averages a tensor, or a dict of tensors across all processes in a process
    group. Objects in all processes will finally have same mean value.

    .. note::

        Nested dicts of tensors are not supported.

    Parameters
    ----------
    t: torch.Tensor or Dict[str, torch.Tensor]
        A tensor or dict of tensors to average across processes.
    """
    if dist.is_initialized():
        if isinstance(t, torch.Tensor):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= get_world_size()
        elif isinstance(t, dict):
            for k in t:
                dist.all_reduce(t[k], op=dist.ReduceOp.SUM)
                t[k] /= dist.get_world_size()


def gpu_mem_usage() -> int:
    r"""
    Return gpu memory usage (in megabytes). If not using GPU, return 0 without
    raising any exceptions.
    """
    if torch.cuda.is_available():
        # This will be in bytes, so we divide by (1024 * 1024).
        return torch.cuda.max_memory_allocated() // 1048576
    else:
        return 0
