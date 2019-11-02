import argparse
import os
import random
import sys
from typing import Iterator

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from viswsl.config import Config
from viswsl.data.datasets import MaskedLanguageModelingDataset
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.factories import VisualStreamFactory, OptimizerFactory
from viswsl.model import ViswslModel
from viswsl.modules.linguistic_stream import LinguisticStream
from viswsl.optim.lr_scheduler import LinearWarmupLinearDecayLR
from viswsl.utils.checkpointing import CheckpointManager
import viswsl.utils.distributed as dist
from viswsl.utils.logging import Timer


# fmt: off
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
    "--serialization-dir", default="checkpoints/experiment",
    help="Path to a directory to serialize config, checkpoints and logs.",
)
parser.add_argument(
    "--log-every", type=int, default=20,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
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
        batch_size=_C.OPTIM.BATCH_SIZE // dist.get_world_size(),
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    val_dataset = MaskedLanguageModelingDataset.from_config(
        _C, vocabulary=vocabulary, tokenizer=tokenizer, split="val"
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )

    # TODO: Make a linguistic stream factory.
    visual_module = VisualStreamFactory.from_config(_C)
    linguistic_module = LinguisticStream.from_config(_C)
    model = ViswslModel(visual_module, linguistic_module).to(device)

    if dist.get_world_size() > 1:
        dist.synchronize()
        model = nn.parallel.DistributedDataParallel(  # type: ignore
            model, device_ids=[device]
        )

    optimizer = OptimizerFactory.from_config(_C, model.parameters())
    lr_scheduler = LinearWarmupLinearDecayLR(
        optimizer,
        total_steps=_C.OPTIM.NUM_ITERATIONS,
        warmup_steps=_C.OPTIM.WARMUP_STEPS,
    )

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
    timer = Timer(
        window_size=_A.log_every, total_iterations=_C.OPTIM.NUM_ITERATIONS
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter: Iterator = iter(train_dataloader)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(1, _C.OPTIM.NUM_ITERATIONS + 1):
        timer.tic()
        # keys: {"image_id", "image", "caption_tokens", "masked_labels"}
        batch = next(train_dataloader_iter)
        for key in batch:
            batch[key] = batch[key].to(device)

        # keys; {"predictions", "loss"}
        output_dict = model(
            batch["image"], batch["caption_tokens"], batch["masked_labels"]
        )
        # Normalize the loss, because gradients are being accumulated (summed)
        # while the loss is averaged across training instances.
        loss = output_dict["loss"].mean() / _C.OPTIM.GRAD_ACCUMULATION_STEPS
        loss.backward()

        if iteration % _C.OPTIM.GRAD_ACCUMULATION_STEPS == 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), _C.OPTIM.CLIP_GRADIENTS
            )
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        timer.toc()

        # Make the master process log loss, lr, time to tensorboard.
        if iteration % _A.log_every == 0 and dist.is_master_process():
            loss = dist.average_across_processes(loss)

            logger.info(f"{timer.stats} | Loss: {loss:.3f}")
            tensorboard_writer.add_scalar("loss", loss, iteration)
            tensorboard_writer.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            tensorboard_writer.add_scalar("avg_time", timer.avg, iteration)
            tensorboard_writer.add_scalar(
                "eta_hours", timer.eta_sec / 3600, iteration
            )
            dist.synchronize()

        # ---------------------------------------------------------------------
        #   VALIDATION
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0 and dist.is_master_process():
            # Remove all accumulated gradients before evaluation.
            torch.set_grad_enabled(False)
            model.eval()

            # Check performance of blind variant of this model too. This is an
            # unfair evaluation, but we keep it only for debugging.
            blind_model = ViswslModel(
                VisualStreamFactory.create("blind"), linguistic_module
            ).to(device)

            normal_val_loss = 0.0
            blind_val_loss = 0.0
            for val_iteration, val_batch in enumerate(val_dataloader):
                output_dict = model(
                    batch["image"], batch["caption_tokens"], batch["masked_labels"]
                )
                blind_output_dict = blind_model(
                    batch["image"], batch["caption_tokens"], batch["masked_labels"]
                )
                normal_val_loss += output_dict["loss"].sum().item()
                blind_val_loss += blind_output_dict["loss"].sum().item()

                if val_iteration == _A.val_batches:
                    normal_val_loss /= _A.val_batches
                    blind_val_loss /= _A.val_batches
                    break

            tensorboard_writer.add_scalar(
                "val/loss_normal", normal_val_loss, iteration
            )
            tensorboard_writer.add_scalar(
                "val/loss_blind", blind_val_loss, iteration
            )

            # -----------------------------------------------------------------
            #   PRINT EXAMPLES
            # -----------------------------------------------------------------
            examples_str = ""
            for tokens, labels, predictions, blind_predictions in zip(
                batch["caption_tokens"][:10],
                batch["masked_labels"][:10],
                output_dict["predictions"][:10],
                blind_output_dict["predictions"][:10],
            ):
                to_strtokens = lambda token_indices: [  # noqa: E731
                    vocabulary.get_token_from_index(t.item())
                    for t in token_indices if t.item() != vocabulary.unk_index
                ]
                tokens = to_strtokens(tokens)
                labels = to_strtokens(labels)
                predictions = to_strtokens(predictions)
                blind_predictions = to_strtokens(blind_predictions)
                # fmt: off
                predictions = [
                    predictions[i] for i in range(len(predictions))
                    if labels[i] != vocabulary.unk_token
                ]
                blind_predictions = [
                    blind_predictions[i] for i in range(len(predictions))
                    if labels[i] != vocabulary.unk_token
                ]
                # fmt: on
                examples_str += f"""
                    Caption tokens      : {tokenizer.detokenize(tokens)}
                    Masked Labels       : {" ".join(labels)}
                    Predictions (normal): {" ".join(predictions)}
                    Predictions (blind) : {" ".join(blind_predictions)}

                    """
            tensorboard_writer.add_text("predictions", examples_str, iteration)

            # Free up memory.
            del blind_model
            checkpoint_manager.step(iteration)
            torch.set_grad_enabled(True)
            model.train()
        # All processes will wait till master process is done logging.
        dist.synchronize()
