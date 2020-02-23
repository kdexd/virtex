import argparse
import os
import random
import sys

from loguru import logger
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from viswsl.config import Config
from viswsl.data import ImageNetDataset, Places205Dataset
from viswsl.factories import PretrainingModelFactory
from viswsl.models.downstream import FeatureExtractor9k, LinearClassifier
from viswsl.utils.checkpointing import CheckpointManager
from viswsl.utils.common import cycle, Timer
import viswsl.utils.distributed as dist


parser = argparse.ArgumentParser(
    description="Train a linear classifier on a pre-trained frozen feature extractor."
)
parser.add_argument(
    "--task", required=True, choices=["imagenet", "places205"],
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
    "--weight-init", choices=["random", "imagenet", "checkpoint"],
    default="checkpoint", help="""How to initialize weights: 'random' initializes
    all weights randomly, 'imagenet' initializes backbone weights from torchvision
    model zoo, and 'checkpoint' loads state dict from `--checkpoint-path`."""
)
parser.add_argument(
    "--log-every", type=int, default=50,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
parser.add_argument(
    "--checkpoint-path",
    help="""Path to load checkpoint and run downstream task evaluation. The
    name of checkpoint file is required to be `model_*.pth`, where * is
    iteration number from which the checkpoint was serialized."""
)
parser.add_argument(
    "--serialization-dir", required=True,
    help="Path to a directory to save checkpoints and log stats."
)
parser.add_argument(
    "--checkpoint-every", type=int, default=5000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
# fmt: on


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()

    # Add an arg in config override if `--weight-init` is imagenet.
    if _A.weight_init == "imagenet":
        _A.config_override.extend(["MODEL.VISUAL.PRETRAINED", True])

    device_id = dist.init_distributed_env(_A.dist_backend) if _A.slurm else -1
    device = torch.device(f"cuda:{device_id}" if device_id != -1 else "cpu")

    _C = Config(_A.config, _A.config_override)
    _DOWNC = _C.DOWNSTREAM.IN_LINEAR

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)

    # Configure our custom logger.
    logger.remove(0)
    logger.add(
        sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
    )
    logger.disable(__name__) if not dist.is_master_process() else None
    os.makedirs(_A.serialization_dir, exist_ok=True)

    # Print config and args.
    logger.info(str(_C))
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER
    # -------------------------------------------------------------------------
    Dataset = ImageNetDataset if _A.task == "imagenet" else Places205Dataset

    train_dataset = Dataset(_DOWNC.DATA_ROOT, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_DOWNC.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        sampler=DistributedSampler(train_dataset, shuffle=True),
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataset = Dataset(_DOWNC.DATA_ROOT, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_DOWNC.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        sampler=DistributedSampler(val_dataset, shuffle=False),
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device)

    # Create mdel and add linear classifier on visual backbone.
    model = PretrainingModelFactory.from_config(_C).to(device)

    if _A.weight_init == "checkpoint":
        model.load_state_dict(
            torch.load(_A.checkpoint_path, map_location=torch.device("cpu"))
        )

    # Backup pretext config and model checkpoint.
    if dist.is_master_process():
        _C.dump(os.path.join(_A.serialization_dir, "pretext_config.yml"))
        torch.save(
            model.state_dict(),
            os.path.join(_A.serialization_dir, "pretext_model.pth"),
        )

    # Wrap our model into `FeatureExtractor9k`, freeze weights and delete model.
    feature_extractor = FeatureExtractor9k(model, normalize_with_bn=True).to(device)
    feature_extractor.eval()
    del model

    # A simple linear layer on top of backbone, `feature_size` is usually 8192.
    classifier = LinearClassifier(
        feature_size=feature_extractor.feature_size["layer4"],
        num_classes=_DOWNC.NUM_CLASSES,
    ).to(device)

    # We don't use factories to create optimizer and scheduler, because they
    # are created differently, and are not customizable for this protocol.
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=_DOWNC.LR,
        momentum=_DOWNC.MOMENTUM,
        weight_decay=_DOWNC.WEIGHT_DECAY,
        nesterov=_DOWNC.NESTEROV,
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=_DOWNC.STEPS, gamma=_DOWNC.GAMMA
    )
    if dist.get_world_size() > 1:
        dist.synchronize()
        # We don't need DDP over model because there's no communication in eval.
        classifier = nn.parallel.DistributedDataParallel(
            classifier, device_ids=[device], find_unused_parameters=True
        )

    # -------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------
    if dist.is_master_process():
        # Only the master process would serialize checkpoints. Keep only recent
        # five checkpoints to save memory.
        checkpoint_manager = CheckpointManager(
            classifier, optimizer, _A.serialization_dir, k_recent=5
        )
        tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)

    # Keep track of (moving) average time per iteration and ETA.
    timer = Timer(window_size=_A.log_every, total_iterations=_DOWNC.NUM_ITERATIONS)

    # A tensor to accumulate loss for logging (and have smooth training curve).
    # Reset to zero every `_A.log_every` iteration.
    train_loss: torch.Tensor = torch.tensor(0.0).to(device)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(1, _DOWNC.NUM_ITERATIONS + 1):
        timer.tic()
        optimizer.zero_grad()
        batch = next(train_dataloader_iter)

        with torch.no_grad():
            # Keep features only from last layer (for this evaluation protocol).
            features = feature_extractor(batch["image"])["layer4"]

        loss = classifier(features, batch["label"])["loss"]
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        lr_scheduler.step()
        timer.toc()

        # ---------------------------------------------------------------------
        #   TENSORBOARD LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.log_every == 0:
            train_loss /= _A.log_every
            dist.average_across_processes(train_loss)

            if dist.is_master_process():
                logger.info(
                    f"{timer.stats} | Loss: {train_loss:.3f} | GPU mem: "
                    f"{torch.cuda.max_memory_allocated() / 1048576} MB"
                )
                tensorboard_writer.add_scalars(
                    "loss", {"train": train_loss}, iteration
                )
            train_loss = torch.zeros_like(train_loss)

        # ---------------------------------------------------------------------
        #   VALIDATION
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            torch.set_grad_enabled(False)
            classifier.eval()

            val_loss: torch.Tensor = torch.tensor(0.0).to(device)
            for val_iteration, val_batch in enumerate(val_dataloader, start=1):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)

                # Keep features only from last layer (for this evaluation protocol).
                features = feature_extractor(batch["image"])["layer4"]
                output_dict = classifier(features, batch["label"])
                val_loss += output_dict["loss"].item()

            # Finally divide val loss by number of val iterations.
            val_loss /= val_iteration
            dist.average_across_processes(val_loss)

            # Get accumulated Top-1 accuracy for logging across GPUs.
            if dist.get_world_size() > 1:
                acc = classifier.module.get_metric(reset=True)
                acc = torch.tensor(acc).to(device)
                dist.average_across_processes(acc)
            else:
                acc = classifier.get_metric(reset=True)

            torch.set_grad_enabled(True)
            classifier.train()

            # Save recent checkpoint and best checkpoint based on accuracy.
            if dist.is_master_process():
                checkpoint_manager.step(iteration, metric=acc)

        # ---------------------------------------------------------------------
        #   TENSORBOARD LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0 and dist.is_master_process():
            logger.info(
                f"Iter: {iteration} | Val loss: {val_loss} | Top-1 Acc: {acc})"
            )
            tensorboard_writer.add_scalars("loss", {"val": val_loss}, iteration)
            tensorboard_writer.add_scalars("metrics", {"top1": acc}, iteration)

        # All processes will wait till master process is done logging.
        dist.synchronize()
