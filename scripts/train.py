import argparse
from collections import Counter
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
from viswsl.data import SentencePieceVocabulary, SentencePieceTokenizer
from viswsl.factories import (
    DatasetFactory, PretrainingModelFactory, OptimizerFactory, LRSchedulerFactory,
)
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
    "--checkpoint-every", type=int, default=2000,
    help="Serialize model to a checkpoint after every these many iterations.",
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
    train_dataset = DatasetFactory.from_config(
        _C, vocabulary=vocabulary, tokenizer=tokenizer, split="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    val_dataset = DatasetFactory.from_config(
        _C, vocabulary=vocabulary, tokenizer=tokenizer, split="val"
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device)

    model = PretrainingModelFactory.from_config(_C).to(device)
    optimizer = OptimizerFactory.from_config(_C, model.named_parameters())
    lr_scheduler = LRSchedulerFactory.from_config(_C, optimizer)

    # Wrap model and optimizer using NVIDIA Apex for mixed precision training.
    # NOTE: Always do this before wrapping model with DistributedDataParallel.
    if _C.MIXED_PRECISION_OPT > 0:
        from apex import amp

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=f"O{_C.MIXED_PRECISION_OPT}"
        )

    if dist.get_world_size() > 1:
        dist.synchronize()
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
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
    timer = Timer(window_size=_A.log_every, total_iterations=_C.OPTIM.NUM_ITERATIONS)

    # Counter to accumulate loss components for logging, this counter is
    # cleared every `_A.log_every` iteration.
    train_loss_counter: Counter = Counter()

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
            if _C.MODEL.NAME == "word_masking":
                output_dict = model(
                    batch["image"],
                    batch["caption_tokens"],
                    batch["caption_lengths"],
                    batch["masked_labels"],
                )
            elif _C.MODEL.NAME == "captioning":
                output_dict = model(
                    batch["image"], batch["caption_tokens"], batch["caption_lengths"]
                )

            train_loss_counter.update(output_dict["loss_components"])

            # Normalize the loss, because gradients are being accumulated
            # (summed) while the loss is averaged across training instances.
            loss = output_dict["loss"] / _C.OPTIM.BATCH_SIZE_MULTIPLIER
            batch_loss += loss.item()

            # Perform dynamic scaling of loss to adjust for mixed precision.
            if _C.MIXED_PRECISION_OPT > 0:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        # Clip norm of gradients before optimizer step.
        torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer)
            if _C.MIXED_PRECISION_OPT > 0
            else model.parameters(),
            _C.OPTIM.CLAMP_GRADIENTS,
        )
        optimizer.step()
        lr_scheduler.step()
        timer.toc()

        # Make the master process log loss, lr, time to tensorboard.
        if iteration % _A.log_every == 0:
            train_loss_dict = {
                k: v / (_A.log_every * _C.OPTIM.BATCH_SIZE_MULTIPLIER)
                for k, v in dict(train_loss_counter).items()
            }
            dist.average_across_processes(train_loss_dict)
            train_loss_counter.clear()

        # ---------------------------------------------------------------------
        #   TENSORBOARD LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.log_every == 0 and dist.is_master_process():
            logger.info(
                f"{timer.stats} | Loss: {batch_loss:.3f} | "
                f"GPU mem: {torch.cuda.max_memory_allocated() / 1048576} MB"
            )
            tensorboard_writer.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            tensorboard_writer.add_scalars("train", train_loss_dict, iteration)

            for name, param in model.named_parameters():
                tensorboard_writer.add_histogram(name, param, iteration)
                tensorboard_writer.add_histogram(
                    name + "_grad", param.grad, iteration
                )

        dist.synchronize()

        # ---------------------------------------------------------------------
        #   VALIDATION
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            torch.set_grad_enabled(False)
            model.eval()

            # Accumulate different val loss components according to the type of
            # pretraining model.
            val_loss_counter: Counter = Counter()

            for val_iteration, val_batch in enumerate(val_dataloader, start=1):
                if _C.MODEL.NAME == "word_masking":
                    args = [
                        batch["image"],
                        batch["caption_tokens"],
                        batch["caption_lengths"],
                        batch["masked_labels"],
                    )
                elif _C.MODEL.NAME == "captioning":
                    output_dict = model(
                        batch["image"],
                        batch["caption_tokens"],
                        batch["caption_lengths"],
                    )

                # This will have a key named "loss_components": these are
                # scalar tensors (mean loss per batch) only for logging.
                output_dict = model(*args)
                val_loss_counter.update(output_dict["loss_components"])

            # Divide each loss component by number of val batches per GPU.
            val_loss_dict = {
                k: v / val_iteration for k, v in dict(val_loss_counter).items()
            }
            dist.average_across_processes(val_loss_dict)
            torch.set_grad_enabled(True)
            model.train()

        # ---------------------------------------------------------------------
        #   TENSORBOARD LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0 and dist.is_master_process():
            checkpoint_manager.step(iteration)
            logger.info(f"Iter: {iteration} | Val loss: {val_loss_dict}")
            tensorboard_writer.add_scalars("val", val_loss_dict, iteration)

            if _C.MODEL.NAME == "word_masking":
                # fmt: off
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
                        Caption tokens : {tokenizer.detokenize(tokens)}
                        Masked Labels  : {" ".join(labels)}
                        Predictions    : {" ".join(predictions)}

                        """
                # fmt: on
                tensorboard_writer.add_text("predictions", examples_str, iteration)

            elif _C.MODEL.NAME == "captioning":
                for tokens, forward_predictions, backward_predictions in zip(
                    batch["caption_tokens"], output_dict["forward_predictions"],
                    output_dict["backward_predictions"]
                ):
                    tokens = to_strtokens(tokens)
                    forward_predictions = to_strtokens(forward_predictions)
                    backward_predictions = to_strtokens(backward_predictions)

                    examples_str += f"""
                        Caption tokens : {tokenizer.detokenize(tokens)}
                        Predictions (f): {tokenizer.detokenize(forward_predictions)}
                        Predictions (b): {tokenizer.detokenize(backward_predictions)}

                        """
                tensorboard_writer.add_text("predictions", examples_str, iteration)
            # fmt: on

        # All processes will wait till master process is done logging.
        dist.synchronize()
