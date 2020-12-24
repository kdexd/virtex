import argparse
import os
import random
import sys

from loguru import logger
import numpy as np
import torch

from virtex.config import Config
import virtex.utils.distributed as dist


def cycle(dataloader, device, start_iteration: int = 0):
    r"""
    A generator to yield batches of data from dataloader infinitely.

    Internally, it sets the ``epoch`` for dataloader sampler to shuffle the
    examples. One may optionally provide the starting iteration to make sure
    the shuffling seed is different and continues naturally.
    """
    iteration = start_iteration

    while True:
        if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            # Set the `epoch` of DistributedSampler as current iteration. This
            # is a way of determinisitic shuffling after every epoch, so it is
            # just a seed and need not necessarily be the "epoch".
            logger.info(f"Beginning new epoch, setting shuffle seed {iteration}")
            dataloader.sampler.set_epoch(iteration)

        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(device)
            yield batch
            iteration += 1


def common_setup(_C: Config, _A: argparse.Namespace, job_type: str = "pretrain"):
    r"""
    Setup common stuff at the start of every pretraining or downstream
    evaluation job, all listed here to avoid code duplication. Basic steps:

    1. Fix random seeds and other PyTorch flags.
    2. Set up a serialization directory and loggers.
    3. Log important stuff such as config, process info (useful during
        distributed training).
    4. Save a copy of config to serialization directory.

    .. note::

        It is assumed that multiple processes for distributed training have
        already been launched from outside. Functions from
        :mod:`virtex.utils.distributed` module ae used to get process info.

    Parameters
    ----------
    _C: virtex.config.Config
        Config object with all the parameters.
    _A: argparse.Namespace
        Command line arguments.
    job_type: str, optional (default = "pretrain")
        Type of job for which setup is to be done; one of ``{"pretrain",
        "downstream"}``.
    """

    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(_C.RANDOM_SEED)
    torch.backends.cudnn.deterministic = _C.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = _C.CUDNN_BENCHMARK
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, f"{job_type}_config.yaml"))

    # Remove default logger, create a logger for each process which writes to a
    # separate log-file. This makes changes in global scope.
    logger.remove(0)
    if dist.get_world_size() > 1:
        logger.add(
            os.path.join(_A.serialization_dir, f"log-rank{RANK}.txt"),
            format="{time} {level} {message}",
        )

    # Add a logger for stdout only for the master process.
    if dist.is_master_process():
        logger.add(
            sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
        )

    # Print process info, config and args.
    logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
    logger.info(str(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))


def common_parser(description: str = "") -> argparse.ArgumentParser:
    r"""
    Create an argument parser some common arguments useful for any pretraining
    or downstream evaluation scripts.

    Parameters
    ----------
    description: str, optional (default = "")
        Description to be used with the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        A parser object with added arguments.
    """
    parser = argparse.ArgumentParser(description=description)

    # fmt: off
    parser.add_argument(
        "--config", metavar="FILE", help="Path to a pretraining config file."
    )
    parser.add_argument(
        "--config-override", nargs="*", default=[],
        help="A list of key-value pairs to modify pretraining config params.",
    )
    parser.add_argument(
        "--serialization-dir", default="/tmp/virtex",
        help="Path to a directory to serialize checkpoints and save job logs."
    )

    group = parser.add_argument_group("Compute resource management arguments.")
    group.add_argument(
        "--cpu-workers", type=int, default=0,
        help="Number of CPU workers per GPU to use for data loading.",
    )
    group.add_argument(
        "--num-machines", type=int, default=1,
        help="Number of machines used in distributed training."
    )
    group.add_argument(
        "--num-gpus-per-machine", type=int, default=0,
        help="""Number of GPUs per machine with IDs as (0, 1, 2 ...). Set as
        zero for single-process CPU training.""",
    )
    group.add_argument(
        "--machine-rank", type=int, default=0,
        help="""Rank of the machine, integer in [0, num_machines). Default 0
        for training with a single machine.""",
    )
    group.add_argument(
        "--dist-url", default=f"tcp://127.0.0.1:23456",
        help="""URL of the master process in distributed training, it defaults
        to localhost for single-machine training.""",
    )
    # fmt: on

    return parser
