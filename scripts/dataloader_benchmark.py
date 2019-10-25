import argparse
import random
import sys
from typing import Iterator

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader

from viswsl.config import Config
from viswsl.data.datasets import MaskedLanguageModelingDataset
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.data.tokenizers import SentencePieceTokenizer
import viswsl.utils.distributed as dist
from viswsl.utils.logging import Timer


# fmt: off
parser = argparse.ArgumentParser(
    description="Benchmark the data-loading speed."
)
parser.add_argument(
    "--config", help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override", nargs="*",
    help="""A sequence of key-value pairs specifying certain config arguments
    (with dict-like nesting) using a dot operator.""",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--cpu-workers", type=int, default=0,
    help="Number of CPU workers per GPU to use for data loading.",
)
parser.add_argument(
    "--dist-backend", default="nccl", choices=["nccl", "gloo"],
    help="torch.distributed backend for distributed training.",
)
parser.add_argument(
    "--slurm", action="store_true",
    help="""Whether using SLURM for launching distributed training processes.
    Setting this flag assumes ignores arguments `--num-gpus-per-machine`,
    `--num-machines`, `--machine-rank` and `--dist-url`. Set `$MASTER_PORT`
    env variable externally for distributed process group communication."""
)
parser.add_argument(
    "--num-gpus-per-machine", type=int, default=0,
    help="Number of GPUs per machine with IDs as 0, 1, 2.. and so on.",
)
parser.add_argument(
    "--num-machines", type=int, default=1,
    help="Number of machines used in distributed training."
)
parser.add_argument(
    "--machine-rank", type=int, default=0,
    help="""Rank of the machine, integer in [0, num_machines). Default 0 for
    training with a single machine.""",
)
parser.add_argument(
    "--dist-url", default=f"tcp://127.0.0.1:23456",
    help="""URL of the master process in distributed training, it defaults to
    localhost for single-machine training.""",
)

parser.add_argument_group("Checkpointing and Logging")
parser.add_argument(
    "--log-every", type=int, default=20,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
# fmt: on


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and
    # _A. This object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config, _A.config_override)

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if _A.slurm:
        device_id = dist.init_distributed_env(_A.dist_backend)
    elif _A.num_gpus_per_machine == 0:
        device_id = -1
    else:
        # TODO (kd): Add an option to use `init_distributed_tcp`.
        device_id = 0
    device = torch.device(f"cuda:{device_id}" if device_id != -1 else "cpu")

    # Disable the logger for all processes except master process to avoid
    # clutter in stdout / stderr / logfile.
    logger.remove(0)
    logger.add(
        sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
    )
    logger.disable(__name__) if not dist.is_master_process() else None

    # Print config and args.
    logger.info(str(_C))
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))

    # -------------------------------------------------------------------------
    #   INSTANTIATE VOCABULARY, TOKENIZER, DATALOADER
    # -------------------------------------------------------------------------
    vocabulary = SentencePieceVocabulary(_C.DATA.VOCABULARY)
    tokenizer = SentencePieceTokenizer(_C.DATA.TOKENIZER)
    train_dataset = MaskedLanguageModelingDataset.from_config(
        _C, vocabulary=vocabulary, tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE // dist.get_world_size(),
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter: Iterator = iter(train_dataloader)

    # Keep track of (moving) average time per iteration and ETA.
    timer = Timer(
        window_size=_A.log_every, total_iterations=_C.OPTIM.NUM_ITERATIONS
    )

    # -------------------------------------------------------------------------
    #   BENCHMARKING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(1, _C.OPTIM.NUM_ITERATIONS + 1):
        timer.tic()
        # keys: {"image_id", "image", "caption_tokens", "masked_labels"}
        batch = next(train_dataloader_iter)
        for key in batch:
            batch[key] = batch[key].to(device)

        # Synchronize every iteratin to record the worst time among processes.
        dist.synchronize()
        timer.toc()

        # Make the master process log loss, lr, time to tensorboard.
        if iteration % _A.log_every == 0 and dist.is_master_process():
            logger.info(timer.stats)
            dist.synchronize()
