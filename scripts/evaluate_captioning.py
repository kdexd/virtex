import argparse
import os
import random
import sys
from typing import Any, Dict, List

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from viswsl.config import Config
from viswsl.factories import (
    TokenizerFactory, DatasetFactory, PretrainingModelFactory,
)
from viswsl.utils.metrics import CocoCaptionsEvaluator


parser = argparse.ArgumentParser(
    description="Evaluate a pre-trained model based on captioning metrics."
)
parser.add_argument(
    "--config", help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override", nargs="*", default=[],
    help="""A sequence of key-value pairs specifying certain config arguments
    (with dict-like nesting) using a dot operator.""",
)
parser.add_argument(
    "--captions",
    default="datasets/coco/annotations/captions_val2017.json",
    help="Path to annotations file with ground truth captions.",
)
parser.add_argument(
    "--gpu-id", type=int, default=0, help="ID of GPU to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=2, help="Number of CPU workers."
)
parser.add_argument(
    "--checkpoint-path", required=True,
    help="""Path to load checkpoint and run captioning evaluation. The
    name of checkpoint file is required to be `checkpoint_*.pth`, where * is
    iteration number from which the checkpoint was serialized."""
)
parser.add_argument(
    "--serialization-dir", default=None,
    help="""Path to a directory to save results log as a Tensorboard event
    file. If not provided, this will be the parent directory of checkpoint."""
)
# fmt: on


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()
    _C = Config(_A.config, _A.config_override)

    # Set random seeds for reproucibility.
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)

    device = torch.device(f"cuda:{_A.gpu_id}" if _A.gpu_id != -1 else "cpu")

    # Configure our custom logger.
    logger.remove(0)
    logger.add(
        sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
    )
    # Set up a serialization directory.
    if not _A.serialization_dir:
        _A.serialization_dir = os.path.dirname(_A.checkpoint_path)
    os.makedirs(_A.serialization_dir, exist_ok=True)

    # Print config and args.
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Tensorboard writer for logging mAP scores.
    tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)
    CHECKPOINT_ITERATION = int(
        os.path.basename(_A.checkpoint_path).split("_")[-1][:-4]
    )

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER
    # -------------------------------------------------------------------------
    tokenizer = TokenizerFactory.from_config(_C)
    val_dataset = DatasetFactory.from_config(_C, tokenizer, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE_PER_GPU,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )
    # Initialize model from a checkpoint.
    model = PretrainingModelFactory.from_config(_C).to(device)
    model.load_state_dict(torch.load(_A.checkpoint_path))
    torch.set_grad_enabled(False)
    model.eval()

    # ---------------------------------------------------------------------
    #   VALIDATION
    # ---------------------------------------------------------------------

    predictions: List[Dict[str, Any]] = []
    for val_iteration, val_batch in enumerate(val_dataloader, start=1):
        for key in val_batch:
            val_batch[key] = val_batch[key].to(device)
        output_dict = model(val_batch)

        # Make a dictionary of predictions in COCO format.
        for image_id, caption in zip(
            val_batch["image_id"], output_dict["predictions"]
        ):
            predictions.append(
                {
                    "image_id": image_id.item(),
                    "caption": tokenizer.decode(caption.tolist),
                }
            )

    # ---------------------------------------------------------------------
    #   CALCULATE AND LOG METRICS
    # ---------------------------------------------------------------------
    metrics = CocoCaptionsEvaluator(_A.captions).evaluate(predictions)
    logger.info(f"Iter: {CHECKPOINT_ITERATION} | Metrics: {metrics}")
    tensorboard_writer.add_scalars("val", metrics, CHECKPOINT_ITERATION)
