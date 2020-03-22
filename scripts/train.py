import argparse
from collections import Counter

from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from viswsl.config import Config
from viswsl.factories import (
    TokenizerFactory, DatasetFactory, PretrainingModelFactory,
    OptimizerFactory, LRSchedulerFactory,
)
from viswsl.utils.checkpointing import CheckpointManager
from viswsl.utils.common import common_parser, common_setup, cycle
import viswsl.utils.distributed as dist
from viswsl.utils.timer import Timer


parser = common_parser(
    description="Train a ViRTex model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--serialization-dir", default="checkpoints/experiment",
    help="Path to a directory to serialize config, checkpoints and logs.",
)
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
        device = torch.device("cpu")
    else:
        # Get the current device as set for current distributed process.
        # Check `launch` function in `viswsl.utils.distributed` module.
        device = torch.cuda.current_device()

    # Create a config object (this will be immutable) and perform common setup
    # such as logging and setting up serialization directory.
    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER
    # -------------------------------------------------------------------------
    tokenizer = TokenizerFactory.from_config(_C)
    train_dataset = DatasetFactory.from_config(_C, tokenizer, split="train")
    val_dataset = DatasetFactory.from_config(_C, tokenizer, split="val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        ),
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        sampler=DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        ),
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )

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

    # Wrap model in DDP if using more than one processes.
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

    # Keep track of time per iteration and ETA.
    timer = Timer(
        last_iteration=start_iteration - 1,
        total_iterations=_C.OPTIM.NUM_ITERATIONS,
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device, start_iteration)

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
                f"GPU mem: {dist.gpu_mem_usage()} MB"
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

            if dist.get_world_size() > 1:
                predstr = model.module.log_predictions(val_batch, tokenizer)
                tensorboard_writer.add_text("predictions", predstr, iteration)

        # All processes will wait till master process is done logging.
        dist.synchronize()


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
