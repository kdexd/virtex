import argparse
import os
from typing import Any, Dict, List

from loguru import logger
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from virtex.config import Config
from virtex.data import CocoCaptionsEvalDataset
from virtex.factories import TokenizerFactory, PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser, common_setup
from virtex.utils.metrics import CocoCaptionsEvaluator


parser = common_parser(
    description="Evaluate a pre-trained model based on captioning metrics."
)
parser.add_argument(
    "--checkpoint-path", required=True,
    help="Path to load checkpoint and run captioning evaluation."
)
parser.add_argument(
    "--serialization-dir", default=None,
    help="""Path to a directory to save results log as a Tensorboard event
    file. If not provided, this will be the parent directory of checkpoint."""
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device (this will be zero here by default).
        device = torch.cuda.current_device()

    # Create a config object (this will be immutable) and perform common setup
    # such as logging and setting up serialization directory.
    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL
    # -------------------------------------------------------------------------
    tokenizer = TokenizerFactory.from_config(_C)
    val_dataset = CocoCaptionsEvalDataset(_C.DATA.ROOT)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    # Initialize model from a checkpoint.
    model = PretrainingModelFactory.from_config(_C).to(device)
    ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)

    # Tensorboard writer for logging CIDEr and SPICE scores.
    tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)

    # ---------------------------------------------------------------------
    #   VALIDATION
    # ---------------------------------------------------------------------

    torch.set_grad_enabled(False)
    model.eval()

    # Make a list of predictions to evaluate.
    predictions: List[Dict[str, Any]] = []

    for val_iteration, val_batch in tqdm(enumerate(val_dataloader, start=1)):
        for key in val_batch:
            val_batch[key] = val_batch[key].to(device)

        # Make a dictionary of predictions in COCO format.
        output_dict = model(val_batch)
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
    #   EVALUATE AND LOG METRICS
    # ---------------------------------------------------------------------
    # Assume ground truth (COCO val2017 annotations) exist.
    gt = os.path.join(_C.DATA.ROOT, "annotations", "captions_val2017.json")

    metrics = CocoCaptionsEvaluator(gt).evaluate(predictions)
    logger.info(f"Iter: {ITERATION} | Metrics: {metrics}")
    tensorboard_writer.add_scalars(
        "metrics/cider", {"forward": metrics["CIDEr"]}, ITERATION
    )
    tensorboard_writer.add_scalars(
        "metrics/spice", {"forward": metrics["SPICE"]}, ITERATION
    )


if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus_per_machine > 1:
        raise ValueError("Using multiple GPUs is not supported for this script.")

    # No distributed training here, just a single process.
    main(_A)
