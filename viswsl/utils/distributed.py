import os
import signal
import threading
from typing import Optional

import ifcfg
import torch
from torch import distributed as dist


EXIT = threading.Event()
EXIT.clear()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly.", flush=True)


signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)
signal.signal(signal.SIGUSR2, _clean_exit_handler)


def init_distributed_for_slurm(
    master_address: Optional[str] = "127.0.0.1",
    master_port: Optional[int] = 8989,
    backend: str = "nccl",
) -> torch.device:
    r"""
    Initialize :mod:`torch.distributed` for SLURM processes. This method picks
    up relevant environment variables set by SLURM controller and uses them to
    initialize the process group.

    Specifically, it looks for the following environment variables:

    1. ``$WORLD_SIZE``: Total number of processes. It is set by SLURM as
       ``$SLURM_NTASKS``. The value is ``nodes * gres``.

    2. ``$WORLD_RANK``: Rank of the process, lies in ``[0, $WORLD_SIZE)``. It is
       set by SLURM as ``$SLURM_PROCID``.

    3. ``$LOCAL_RANK``: If we are using multi-node multi-GPU, then this will be
       the GPU ID on the current node (on which the ``$WORLD_RANK`` process is
       initialized). For example: ``--nodes 2 --gres gpu:2`` would initialize
       four processes on two nodes, each with ``$WORLD_RANK: 0, 1, 2, 3`` and
       ``$LOCAL_RANK: 0, 1, 0, 1``.

    Note
    ----
    Use NCCL Backend for GPU training, GLOO backend for debugging.

    Parameters
    ----------
    master_address: str, optional (default = "127.0.0.1")
        IP Address of the master node for distributed training. For single-node
        multi-GPU training, this can be ``"localhost"`` or ``"127.0.0.1"``.
    master_port: int, optional (default = 8989)
        A port on the master node for communication between processes. Make
        sure this port is free, and does not conflict with anything (such as
        Tensorboard).
    backend: str, optional (default = "nccl")
        Backend for :mod:`torch.distributed`, either "gloo" or "nccl".

    """
    os.environ["NCCL_SOCKET_IFNAME"] = ifcfg.default_interface()["device"]

    # Rank of the process in our torch.distributed process group.
    # For SLURM, it is lies in [0, `nodes * gres`).
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1))
    )

    tcp_store = dist.TCPStore(
        master_address, master_port, world_size, world_rank == 0
    )
    dist.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0))
    )

    device = torch.device(f"cuda:{local_rank}")
    return device
