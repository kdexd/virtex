import argparse
import random
import sys

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader

from viswsl.config import Config
from viswsl.data import (
    ImageCaptionDataset, SentencePieceVocabulary, SentencePieceTokenizer
)
import viswsl.utils.distributed as dist
from viswsl.utils.common import cycle, Timer


# fmt: off
parser = argparse.ArgumentParser(
    description="Benchmark the data-loading speed."
)
parser.add_argument(
    "--config", help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override", nargs="*", default=[],
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
    else:
        device_id = -1
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
    train_dataset = ImageCaptionDataset.from_config(
        _C, vocabulary=vocabulary, tokenizer=tokenizer, split="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    train_dataloader_iter = cycle(train_dataloader, device)

    # Keep track of (moving) average time per iteration and ETA.
    timer = Timer(
        window_size=_A.log_every, total_iterations=_C.OPTIM.NUM_ITERATIONS
    )

    # -------------------------------------------------------------------------
    #   BENCHMARKING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(1, _C.OPTIM.NUM_ITERATIONS + 1):
        timer.tic()
        batch = next(train_dataloader_iter)

        # Synchronize every iteratin to record the worst time among processes.
        dist.synchronize()
        timer.toc()

    #     examples_str = ""
    #     for tokens, labels in zip(
    #         batch["caption_tokens"], batch["masked_labels"],
    #     ):
    #         to_strtokens = lambda token_indices: [  # noqa: E731
    #             vocabulary.get_token_from_index(t.item())
    #             for t in token_indices if t.item() != vocabulary.unk_index
    #         ]
    #         tokens = to_strtokens(tokens)
    #         labels = to_strtokens(labels)

    #         examples_str += f"""
    #             Caption tokens      : {tokenizer.detokenize(tokens)}
    #             Masked Labels       : {" ".join(labels)}

    #             """
    #     break
    # print(examples_str)
