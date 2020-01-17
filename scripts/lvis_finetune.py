# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import argparse
import os
import random
import re
import sys

from loguru import logger
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import detectron2 as d2
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import LVISEvaluator

from viswsl.config import Config
from viswsl.factories import PretrainingModelFactory
import viswsl.utils.distributed as dist


parser = argparse.ArgumentParser(description="LVIS Fine-tuning with detectron2")
# fmt: off
parser.add_argument(
    "--config", required=True,
    help="""Path to a config file used to train the model whose checkpoint will
    be loaded (not Detectron2 config)."""
)
parser.add_argument(
    "--cpu-workers", type=int, default=2, help="Number of CPU workers."
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
    "--checkpoint-path", required=True,
    help="""Path to load checkpoint and run downstream task evaluation. The
    name of checkpoint file is required to be `checkpoint_*.pth`, where * is
    iteration number from which the checkpoint was serialized."""
)
parser.add_argument(
    "--serialization-dir", default=None,
    help="""Path to a directory to save results log as a Tensorboard event
    file. If not provided, this will be the parent directory of checkpoint."""
)
parser.add_argument(
    "--checkpoint-every", type=int, default=10000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
# fmt: on


def get_d2_config(_C: Config, _A: argparse.Namespace):
    _D2C = d2.config.get_cfg()

    # Override some default values based on our config file.
    _D2C.merge_from_file(_C.DOWNSTREAM.LVIS.D2_CONFIG)

    # Set ResNet depth to override in Detectron2's config.
    _D2C.MODEL.RESNETS.DEPTH = int(
        re.search(r"resnet(\d+)", _C.MODEL.VISUAL.NAME).group(1)
        if "torchvision" in _C.MODEL.VISUAL.NAME
        else re.search(r"_R_(\d+)", _C.MODEL.VISUAL.NAME).group(1)
        if "detectron2" in _C.MODEL.VISUAL.NAME
        else 0
    )
    _D2C.SEED = _C.RANDOM_SEED
    _D2C.MODEL.RESNETS.NORM = _C.DOWNSTREAM.LVIS.NORM_LAYER

    _D2C.DATALOADER.NUM_WORKERS = _A.cpu_workers
    _D2C.SOLVER.EVAL_PERIOD = _A.checkpoint_every
    _D2C.SOLVER.CHECKPOINT_PERIOD = _A.checkpoint_every

    _D2C.SOLVER.BASE_LR = 0.0025 * dist.get_world_size()
    _D2C.SOLVER.IMS_PER_BATCH = 2 * dist.get_world_size()

    # Set ImageNet pixel mean and std for normalization (BGR).
    _D2C.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    _D2C.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    _D2C.INPUT.FORMAT = "RGB"

    return _D2C


class LvisFinetuneTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return LVISEvaluator(dataset_name, cfg, True, output_folder)


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()

    device_id = dist.init_distributed_env(_A.dist_backend) if _A.slurm else -1
    device = torch.device(f"cuda:{device_id}" if device_id != -1 else "cpu")

    # Create config with default values, then override from config file.
    _C = Config(_A.config)

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

    # Print config and args.
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))

    CHECKPOINT_ITERATION = int(
        os.path.basename(_A.checkpoint_path).split("_")[-1][:-4]
    )
    # Set up a serialization directory.
    if not _A.serialization_dir:
        _A.serialization_dir = os.path.dirname(_A.checkpoint_path)
    os.makedirs(
        os.path.join(_A.serialization_dir, f"lvis_{CHECKPOINT_ITERATION}"),
        exist_ok=True
    )
    # Tensorboard writer for logging mAP scores.
    tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)

    # -------------------------------------------------------------------------
    #   DETECTRON2 CONFIG AND TRAINER
    # -------------------------------------------------------------------------
    _D2C = get_d2_config(_C, _A)
    _D2C.OUTPUT_DIR = os.path.join(
        _A.serialization_dir, f"lvis_{CHECKPOINT_ITERATION}"
    )
    _D2C.freeze()
    logger.info(_D2C)

    # Initialize from a checkpoint, but only keep the visual module.
    model = PretrainingModelFactory.from_config(_C).to(device)
    model.load_state_dict(torch.load(_A.checkpoint_path))
    d2_weights = model.visual.detectron2_backbone_state_dict()
    del model

    trainer = LvisFinetuneTrainer(_D2C)
    trainer.checkpointer._load_model(d2_weights)
    trainer.train()
