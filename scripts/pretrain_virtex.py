import argparse
from collections import Counter
from typing import Any

from loguru import logger
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from virtex.config import Config
from virtex.factories import (
    PretrainingDatasetFactory, PretrainingModelFactory, OptimizerFactory,
    LRSchedulerFactory,
)
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser, common_setup, cycle
import virtex.utils.distributed as dist
from virtex.utils.timer import Timer


parser = common_parser(
    description="Train a VirTex model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
)
group.add_argument(
    "--checkpoint-every", type=int, default=2000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
group.add_argument(
    "--log-every", type=int, default=20,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device: Any = torch.device("cpu")
    else:
        # Get the current device as set for current distributed process.
        # Check `launch` function in `virtex.utils.distributed` module.
        device = torch.cuda.current_device()

    # Create a config object (this will be immutable) and perform common setup
    # such as logging and setting up serialization directory.
    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER, SCHEDULER
    # -------------------------------------------------------------------------
    train_dataset = PretrainingDatasetFactory.from_config(_C, split="train")
    val_dataset = PretrainingDatasetFactory.from_config(_C, split="val")

    # Make `DistributedSampler`s to shard datasets across GPU processes.
    # Skip this if training on CPUs.
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)  # type: ignore
        if _A.num_gpus_per_machine > 0
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False)  # type: ignore
        if _A.num_gpus_per_machine > 0
        else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE // dist.get_world_size(),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE // dist.get_world_size(),
        sampler=val_sampler,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )

    model = PretrainingModelFactory.from_config(_C).to(device)
    optimizer = OptimizerFactory.from_config(_C, model.named_parameters())
    scheduler = LRSchedulerFactory.from_config(_C, optimizer)

    # -------------------------------------------------------------------------
    #   BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------

    # Create a gradient scaler for automatic mixed precision.
    scaler = amp.GradScaler(enabled=_C.AMP)

    # Load checkpoint to resume training if specified.
    if _A.resume_from is not None:
        start_iteration = CheckpointManager(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        ).load(_A.resume_from)
    else:
        start_iteration = 0

    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device, start_iteration)

    # Wrap model in DDP if using more than one processes.
    if dist.get_world_size() > 1:
        dist.synchronize()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # Keep track of time per iteration and ETA.
    timer = Timer(
        start_from=start_iteration + 1, total_iterations=_C.OPTIM.NUM_ITERATIONS
    )
    # Create tensorboard writer and checkpoint manager (only in master process).
    if dist.is_master_process():
        tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)
        tensorboard_writer.add_text("config", f"```\n{_C}\n```")

        checkpoint_manager = CheckpointManager(
            _A.serialization_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, _C.OPTIM.NUM_ITERATIONS + 1):
        timer.tic()
        optimizer.zero_grad()
        batch = next(train_dataloader_iter)

        with amp.autocast(enabled=_C.AMP):
            output_dict = model(batch)
            loss = output_dict["loss"]

        scaler.scale(loss).backward()

        # First clip norm of gradients, and then perform optimizer step.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), _C.OPTIM.CLIP_GRAD_NORM)
        scaler.step(optimizer)

        scaler.update()
        scheduler.step()
        timer.toc()

        # ---------------------------------------------------------------------
        #   LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.log_every == 0:
            logger.info(
                f"{timer.stats} [Loss {loss:.3f}] [GPU {dist.gpu_mem_usage()} MB]"
            )
            if dist.is_master_process():
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

            # All processes will wait till master process is done serializing.
            dist.synchronize()

            torch.set_grad_enabled(False)
            model.eval()

            # Accumulate different val loss components according to the type of
            # pretraining model.
            val_loss_counter: Counter = Counter()

            for val_iteration, val_batch in enumerate(val_dataloader, start=1):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)
                output_dict = model(val_batch)

                val_loss_counter.update(output_dict["loss_components"])

            # Divide each loss component by number of val batches per GPU.
            val_loss_dict = {
                k: v / val_iteration for k, v in dict(val_loss_counter).items()
            }
            dist.average_across_processes(val_loss_dict)
            torch.set_grad_enabled(True)
            model.train()

            logger.info(f"Iteration: {iteration} [Val loss: {val_loss_dict}]")
            if dist.is_master_process():
                tensorboard_writer.add_scalars("val", val_loss_dict, iteration)


if __name__ == "__main__":
    _A = parser.parse_args()

    if _A.num_gpus_per_machine == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus_per_machine,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            args=(_A, ),
        )
