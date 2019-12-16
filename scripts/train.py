import argparse
import os
import random
import sys

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from viswsl.config import Config
from viswsl.data.datasets import MaskedLanguageModelingDataset
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.factories import (
    VisualStreamFactory, TextualStreamFactory, OptimizerFactory,
    LRSchedulerFactory,
)
from viswsl.model import ViswslModel
from viswsl.utils.checkpointing import CheckpointManager
from viswsl.utils.common import cycle, Timer
import viswsl.utils.distributed as dist


parser = argparse.ArgumentParser(
    description="Train a CNN+Transformer model on masked language modeling."
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
    Set `$MASTER_PORT` env variable externally for distributed process group
    communication."""
)

parser.add_argument_group("Checkpointing and Logging")
parser.add_argument(
    "--serialization-dir", default="checkpoints/experiment",
    help="Path to a directory to serialize config, checkpoints and logs.",
)
parser.add_argument(
    "--log-every", type=int, default=20,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
parser.add_argument(
    "--checkpoint-after", type=int, default=20000,
    help="Start checkpointing after this iteration.",
)
parser.add_argument(
    "--checkpoint-every", type=int, default=2000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
parser.add_argument(
    "--val-batches", type=int, default=100,
    help="Number of batches to perform validation for."
)
# fmt: on


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()

    if _A.slurm:
        device_id = dist.init_distributed_env(_A.dist_backend)
    else:
        # TODO (kd): Add an option to use `init_distributed_tcp`.
        device_id = 0
    device = torch.device(f"cuda:{device_id}" if device_id != -1 else "cpu")

    # Create a config with default values, then override from config file, and
    # _A. This object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config, _A.config_override)

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

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
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER
    # -------------------------------------------------------------------------
    vocabulary = SentencePieceVocabulary(_C.DATA.VOCABULARY)
    tokenizer = SentencePieceTokenizer(_C.DATA.TOKENIZER)
    train_dataset = MaskedLanguageModelingDataset.from_config(
        _C, vocabulary=vocabulary, tokenizer=tokenizer, split="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    val_dataset = MaskedLanguageModelingDataset.from_config(
        _C, vocabulary=vocabulary, tokenizer=tokenizer, split="val"
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )

    model = ViswslModel(
        VisualStreamFactory.from_config(_C), TextualStreamFactory.from_config(_C)
    ).to(device)

    if dist.get_world_size() > 1:
        dist.synchronize()
        model = nn.parallel.DistributedDataParallel(  # type: ignore
            model, device_ids=[device]
        )

    optimizer = OptimizerFactory.from_config(_C, model.named_parameters())
    lr_scheduler = LRSchedulerFactory.from_config(_C, optimizer)

    # -------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------
    if dist.is_master_process():
        # Only the master process would serialize checkpoints.
        checkpoint_manager = CheckpointManager(
            model, optimizer, _A.serialization_dir
        )
        # Tensorboard writer for logging training curves. Only the master
        # process will log events to tensorboard to avoid clutter.
        tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)
        tensorboard_writer.add_text("config", str(_C))

    # Keep track of (moving) average time per iteration and ETA.
    timer = Timer(window_size=_A.log_every, total_iterations=_C.OPTIM.NUM_ITERATIONS)
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(1, _C.OPTIM.NUM_ITERATIONS + 1):
        timer.tic()
        optimizer.zero_grad()

        # Simulate a larger batch size (mostly due to GPU constraints).
        batch_loss = torch.tensor(0.0, device=device)
        for _ in range(_C.OPTIM.BATCH_SIZE_MULTIPLIER):
            batch = next(train_dataloader_iter)

            # keys; {"predictions", "loss"}
            output_dict = model(
                batch["image"], batch["caption_tokens"], batch["masked_labels"]
            )
            # Normalize the loss, because gradients are being accumulated
            # (summed) while the loss is averaged across training instances.
            loss = output_dict["loss"].mean() / _C.OPTIM.BATCH_SIZE_MULTIPLIER
            batch_loss += loss.item()
            loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(
                    min=-_C.OPTIM.CLAMP_GRADIENTS, max=_C.OPTIM.CLAMP_GRADIENTS
                )
        optimizer.step()
        lr_scheduler.step()
        timer.toc()

        # Make the master process log loss, lr, time to tensorboard.
        if iteration % _A.log_every == 0:
            batch_loss = dist.average_across_processes(batch_loss)

            if dist.is_master_process():
                logger.info(f"{timer.stats} | Loss: {batch_loss:.3f}")
                tensorboard_writer.add_scalar("loss", batch_loss, iteration)
                tensorboard_writer.add_scalar(
                    "learning_rate", optimizer.param_groups[0]["lr"], iteration
                )
            dist.synchronize()

        # ---------------------------------------------------------------------
        #   VALIDATION
        # ---------------------------------------------------------------------
        if iteration >= _A.checkpoint_after and iteration % _A.checkpoint_every == 0:
            torch.set_grad_enabled(False)
            model.eval()

            val_loss = torch.tensor(0.0, device=device)
            for val_iteration, val_batch in enumerate(val_dataloader):
                output_dict = model(
                    batch["image"], batch["caption_tokens"], batch["masked_labels"]
                )
                val_loss += output_dict["loss"].mean().item()

                if val_iteration == _A.val_batches // dist.get_world_size():
                    val_loss /= _A.val_batches // dist.get_world_size()
                    break

            dist.average_across_processes(val_loss)
            torch.set_grad_enabled(True)
            model.train()

        # ---------------------------------------------------------------------
        #   TENSORBOARD LOGGING
        # ---------------------------------------------------------------------
        if (
            iteration >= _A.checkpoint_after
            and iteration % _A.checkpoint_every == 0
            and dist.is_master_process()
        ):
            # fmt: off
            logger.info(
                f"Iter: {iteration} | Val loss- masked_lm: {val_loss:.3f} "
            )
            tensorboard_writer.add_scalar(
                "val/masked_lm_loss", val_loss, iteration
            )

            examples_str = ""
            for tokens, labels, predictions in zip(
                batch["caption_tokens"], batch["masked_labels"],
                output_dict["predictions"]
            ):
                # Keep predictions only from [MASK]ed positions.
                predictions = [
                    predictions[i] for i in range(len(predictions))
                    if labels[i] != vocabulary.unk_index
                ]
                to_strtokens = lambda token_indices: [  # noqa: E731
                    vocabulary.get_token_from_index(t.item())
                    for t in token_indices if t.item() != vocabulary.unk_index
                ]
                tokens = to_strtokens(tokens)
                labels = to_strtokens(labels)
                predictions = to_strtokens(predictions)

                examples_str += f"""
                    Caption tokens      : {tokenizer.detokenize(tokens)}
                    Masked Labels       : {" ".join(labels)}
                    Predictions (normal): {" ".join(predictions)}

                    """
            # fmt: on
            tensorboard_writer.add_text("predictions", examples_str, iteration)
            checkpoint_manager.step(iteration)

        # All processes will wait till master process is done logging.
        dist.synchronize()
