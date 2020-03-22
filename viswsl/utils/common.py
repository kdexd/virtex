import argparse
import os
import random
import sys

from loguru import logger
import numpy as np
import torch

from viswsl.config import Config
import viswsl.utils.distributed as dist


def cycle(dataloader, device, start_iteration: int = 0):
    r"""
    A generator which yields batch from dataloader perpetually.
    This is done so because we train for a fixed number of iterations, and do
    not have the notion of 'epochs'.

    Internally, it sets the ``epoch`` for dataloader sampler to shuffle the
    examples. One may optionally provide the starting iteration to make sure
    the shuffling seed is difference and continues naturally.
    """
    iteration = start_iteration

    while True:
        # Set the `epoch` of sampler as current iteration. This is just for
        # determinisitic shuffling after every epoch, so it is just a seed and
        # need not necessarily be the "epoch".
        logger.info(f"Beginning new epoch, setting shuffle seed {iteration}")
        dataloader.sampler.set_epoch(iteration)

        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(device)
            yield batch
            iteration += 1


def common_setup(_C: Config, _A: argparse.Namespace):
    r"""
    Setup common stuff at the start of every job, all listed here to avoid
    code duplication. Basic steps include::

        1. Fix random seeds and other PyTorch flags.
        2. Set up a serialization directory and loggers.
        3. Log important stuff such as config, process info (useful during
           distributed training).
        4. Save a copy of config to serialization directory.

    .. note::

        It is assumed that multiple processes for distributed training have
        already been launched from outside and functions from
        :mod:`viswsl.util.distributed` will return process info.

    Parameters
    ----------
    _C: viswsl.config.Config
    _A: argparse.Namespace
    """

    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(_C.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

    # Remove default logger, create a logger for each process which writes to a
    # separate log-file. This makes changes in global scope.
    logger.remove(0)
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
    Add some common arguments useful for any training/validation scripts. These
    are mainly related to config file path, config override args, and other
    args related to CPU/GPU resources and distributed training.

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
    group = parser.add_argument_group("ViRTex pretraining config arguments.")
    group.add_argument(
        "--config", metavar="FILE",
        help="Path to a config file with necessary config params."
    )
    group.add_argument(
        "--config-override", nargs="*", default=[],
        help="A list of key-value pairs to modify config params.",
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
