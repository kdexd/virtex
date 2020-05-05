import argparse
from collections import Counter
import os

from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from virtex.config import Config
from virtex.factories import (
    DownstreamDatasetFactory,
    PretrainingModelFactory,
    OptimizerFactory,
    LRSchedulerFactory,
)
from virtex.models.downstream import FeatureExtractor, LinearClassifier
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser, common_setup, cycle
import virtex.utils.distributed as dist
from virtex.utils.timer import Timer


# fmt: off
parser = common_parser(
    description="Train a linear classifier on a pre-trained frozen feature extractor."
)
group = parser.add_argument_group("Downstream config arguments.")
group.add_argument(
    "--down-config", metavar="FILE", help="Path to a downstream config file."
)
group.add_argument(
    "--down-config-override", nargs="*", default=[],
    help="A list of key-value pairs to modify downstream config params.",
)

parser.add_argument_group("Checkpointing and Logging")
parser.add_argument(
    "--weight-init", choices=["random", "imagenet", "torchvision", "checkpoint"],
    default="checkpoint", help="""How to initialize weights:
        1. 'random' initializes all weights randomly
        2. 'imagenet' initializes backbone weights from torchvision model zoo
        3. {'torchvision', 'checkpoint'} load state dict from --checkpoint-path
            - with 'torchvision', state dict would be from PyTorch's training
              script.
            - with 'checkpoint' it should be for our full pretrained model."""
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
    "--checkpoint-every", type=int, default=5000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device as set for current distributed process.
        # Check `launch` function in `virtex.utils.distributed` module.
        device = torch.cuda.current_device()

    # Create a downstream config object (this will be immutable) and perform
    # common setup such as logging and setting up serialization directory.
    _DOWNC = Config(_A.down_config, _A.down_config_override)
    common_setup(_DOWNC, _A, job_type="downstream")

    # Create a (pretraining) config object and backup in serializaion directory.
    _C = Config(_A.config, _A.config_override)
    _C.dump(os.path.join(_A.serialization_dir, "pretrain_config.yaml"))

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER, SCHEDULER
    # -------------------------------------------------------------------------
    train_dataset = DownstreamDatasetFactory.from_config(_DOWNC, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_DOWNC.OPTIM.BATCH_SIZE // dist.get_world_size(),
        num_workers=_A.cpu_workers,
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        ),
        drop_last=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataset = DownstreamDatasetFactory.from_config(_DOWNC, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_DOWNC.OPTIM.BATCH_SIZE // dist.get_world_size(),
        num_workers=_A.cpu_workers,
        sampler=DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        ),
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device)

    # Initialize from a checkpoint, but only keep the visual module.
    pretrained_model = PretrainingModelFactory.from_config(_C)

    # Load weights according to the init method, do nothing for `random`, and
    # `imagenet` is already taken care of.
    if _A.weight_init == "checkpoint":
        CheckpointManager(model=pretrained_model).load(_A.checkpoint_path)
    elif _A.weight_init == "torchvision":
        # Keep strict=False because this state dict may have weights for
        # last fc layer.
        pretrained_model.visual.cnn.load_state_dict(
            torch.load(_A.checkpoint_path, map_location="cpu")["state_dict"],
            strict=False,
        )

    feature_extractor = FeatureExtractor(pretrained_model, layer_name="avgpool")
    feature_extractor = feature_extractor.to(device).eval()

    # Instantiate a linear classifier for ImageNet on top of feature extractor.
    model = LinearClassifier(feature_size=2048, num_classes=1000,).to(device)
    del pretrained_model

    optimizer = OptimizerFactory.from_config(_DOWNC, model.named_parameters())
    lr_scheduler = LRSchedulerFactory.from_config(_DOWNC, optimizer)

    if dist.get_world_size() > 1:
        dist.synchronize()
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )

    # -------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------
    if dist.is_master_process():
        checkpoint_manager = CheckpointManager(
            _A.serialization_dir, model=model, optimizer=optimizer
        )
        tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)

    # Keep track of time per iteration and ETA.
    timer = Timer(start_from=1, total_iterations=_DOWNC.OPTIM.NUM_ITERATIONS)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(1, _DOWNC.OPTIM.NUM_ITERATIONS + 1):
        timer.tic()
        optimizer.zero_grad()
        batch = next(train_dataloader_iter)

        with torch.no_grad():
            features = feature_extractor(batch["image"])

        output_dict = model(features, batch["label"])
        loss = output_dict["loss"]

        loss.backward()
        optimizer.step()
        lr_scheduler.step(iteration)
        timer.toc()

        if iteration % _A.log_every == 0 and dist.is_master_process():
            logger.info(
                f"{timer.stats} | Loss: {loss:.3f} | GPU: {dist.gpu_mem_usage()} MB"
            )
            tensorboard_writer.add_scalar("imagenet/train_loss", loss, iteration)
            tensorboard_writer.add_scalar(
                "imagenet/learning_rate",
                optimizer.param_groups[0]["lr"],
                iteration,
            )

        # ---------------------------------------------------------------------
        #   VALIDATION
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            torch.set_grad_enabled(False)
            model.eval()

            total_val_loss = torch.tensor(0.0).to(device)

            for val_iteration, val_batch in enumerate(val_dataloader, start=1):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)

                features = feature_extractor(val_batch["image"])
                output_dict = model(features, val_batch["label"])
                total_val_loss += output_dict["loss"]

            # Divide each loss component by number of val batches per GPU.
            total_val_loss = total_val_loss / val_iteration
            dist.average_across_processes(total_val_loss)

            # Get accumulated Top-1 accuracy for logging across GPUs.
            if dist.get_world_size() > 1:
                acc = model.module.get_metric(reset=True)
                acc = {k: torch.tensor(v).to(device) for k, v in acc.items()}
                dist.average_across_processes(acc)
            else:
                acc = model.get_metric(reset=True)

            torch.set_grad_enabled(True)
            model.train()

            # Save recent checkpoint and best checkpoint based on accuracy.
            if dist.is_master_process():
                checkpoint_manager.step(iteration)

        if iteration % _A.checkpoint_every == 0 and dist.is_master_process():
            logger.info(f"Iter: {iteration} | Accuracies: {acc})")
            tensorboard_writer.add_scalar(
                "imagenet/val_loss", total_val_loss, iteration
            )
            # This name scoping will result in Tensorboard displaying all metrics
            # (VOC07, caption, etc.) together.
            tensorboard_writer.add_scalars(f"metrics/imagenet", acc, iteration)

        # All processes will wait till master process is done logging.
        dist.synchronize()


if __name__ == "__main__":
    _A = parser.parse_args()

    # Add an arg in config override if `--weight-init` is imagenet.
    if _A.weight_init == "imagenet":
        _A.config_override.extend(["MODEL.VISUAL.PRETRAINED", True])

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
            args=(_A,),
        )
