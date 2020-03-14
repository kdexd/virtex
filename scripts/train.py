import argparse
from collections import Counter
import os
import random
import sys

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from viswsl.config import Config
from viswsl.factories import (
    TokenizerFactory, DatasetFactory, PretrainingModelFactory,
    OptimizerFactory, LRSchedulerFactory,
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
    "--resume-from", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
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

    device_id = dist.init_distributed_env(_A.dist_backend) if _A.slurm else -1
    device = torch.device(f"cuda:{device_id}" if device_id != -1 else "cpu")

    # Create a config with default values, then override from config file, and
    # _A. This object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config, _A.config_override)

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)

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
    tokenizer = TokenizerFactory.from_config(_C)
    train_dataset = DatasetFactory.from_config(_C, tokenizer, split="train")
    val_dataset = DatasetFactory.from_config(_C, tokenizer, split="val")

    # HACK: set DistributedSampler only for `instance_classification`. Remove
    # this later.
    if _C.MODEL.NAME == "instance_classification":
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device)

    model = PretrainingModelFactory.from_config(_C).to(device)
    optimizer = OptimizerFactory.from_config(_C, model.named_parameters())
    scheduler = LRSchedulerFactory.from_config(_C, optimizer)

    # Wrap model and optimizer using NVIDIA Apex for mixed precision training.
    # NOTE: Always do this before wrapping model with DistributedDataParallel.
    if _C.FP16_OPT > 0:
        from apex import amp

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=f"O{_C.FP16_OPT}"
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
            _A.serialization_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # Tensorboard writer for logging training curves. Only the master
        # process will log events to tensorboard to avoid clutter.
        tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)

        # Add whitespace before line breaks because Tensorboard takes Markdown.
        tensorboard_writer.add_text("config", str(_C).replace("\n", "  \n"))

    # Load checkpoint to resume training if specified.
    if _A.resume_from is not None:
        start_iteration = CheckpointManager(
            model=model, optimizer=optimizer, scheduler=scheduler
        ).load(_A.resume_from)
    else:
        start_iteration = 0

    # Keep track of (moving) average time per iteration and ETA.
    timer = Timer(
        window_size=_A.log_every,
        total_iterations=_C.OPTIM.NUM_ITERATIONS,
        last_iteration=start_iteration - 1,
    )

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration, _C.OPTIM.NUM_ITERATIONS):
        timer.tic()
        optimizer.zero_grad()

        batch_loss = torch.tensor(0.0, device=device)

        batch = next(train_dataloader_iter)
        output_dict = model(batch)

        loss = output_dict["loss"]
        batch_loss += loss.item()

        # Perform dynamic scaling of loss to adjust for mixed precision.
        if _C.FP16_OPT > 0:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # Clip norm of gradients before optimizer step.
        torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer) if _C.FP16_OPT > 0 else model.parameters(),
            _C.OPTIM.CLIP_GRAD_NORM,
        )
        optimizer.step()
        scheduler.step(iteration)
        timer.toc()

        # ---------------------------------------------------------------------
        #   TENSORBOARD LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.log_every == 0 and dist.is_master_process():
            logger.info(
                f"{timer.stats} | Loss: {batch_loss:.3f} | "
                f"GPU mem: {torch.cuda.max_memory_allocated() / 1048576} MB"
            )
            tensorboard_writer.add_scalars(
                "learning_rate",
                {
                    "visual": optimizer.param_groups[0]["lr"],
                    "common": optimizer.param_groups[-1]["lr"],
                },
                iteration,
            )
            tensorboard_writer.add_scalars(
                "train", output_dict["loss_components"], iteration
            )

        # ---------------------------------------------------------------------
        #   VALIDATION
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            if dist.is_master_process():
                checkpoint_manager.step(iteration)

            torch.set_grad_enabled(False)
            model.eval()

            # Accumulate different val loss components according to the type of
            # pretraining model.
            val_loss_counter: Counter = Counter()

            for val_iteration, val_batch in enumerate(val_dataloader, start=1):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)
                output_dict = model(val_batch)

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
            logger.info(f"Iter: {iteration} | Val loss: {val_loss_dict}")
            tensorboard_writer.add_scalars("val", val_loss_dict, iteration)

            if dist.get_world_size() > 1 and hasattr(
                model.module, "log_predictions"
            ):
                predstr = model.module.log_predictions(val_batch, tokenizer)
                tensorboard_writer.add_text("predictions", predstr, iteration)

        # All processes will wait till master process is done logging.
        dist.synchronize()
