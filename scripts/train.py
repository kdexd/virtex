import argparse
import os
from typing import Iterator
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from viswsl.config import Config
from viswsl.data.datasets import MaskedLanguageModelingDataset
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.factories import OptimizerFactory
from viswsl.model import ViswslModel
from viswsl.modules.linguistic_stream import LinguisticStream
from viswsl.modules.visual_stream import VisualStream
from viswsl.optim.lr_scheduler import LinearWarmupLinearDecayLR
from viswsl.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser(
    description="""Train only the linguistic stream (transformer) on masked
    language modeling pretext task."""
)
parser.add_argument(
    "--config",
    required=True,
    help="Path to a config file with all configuration parameters.",
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="""A sequence of key-value pairs specifying certain config arguments
    (with dict-like nesting) using a dot operator. The actual config will be
    updated and recorded in the serialization directory.""",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--gpu-ids",
    required=True,
    nargs="+",
    type=int,
    help="List of GPU IDs to use (-1 for CPU).",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=0,
    help="Number of CPU workers to use for data loading.",
)

parser.add_argument_group("Checkpointing related arguments.")
parser.add_argument(
    "--serialization-dir",
    default="checkpoints/experiment",
    help="""Path to a (non-existent) directory for serializing config, model
    checkpoints and tensorboard logs.""",
)
parser.add_argument(
    "--checkpoint-every",
    default=1000,
    type=int,
    help="Serialize model to a checkpoint after every these many iterations.",
)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and
    # _A. This object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config, _A.config_override)

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device according to specified GPU ids.
    device = torch.device(
        f"cuda:{_A.gpu_ids[0]}" if _A.gpu_ids[0] >= 0 else "cpu"
    )
    if len(_A.gpu_ids) > 1:
        warnings.warn(
            f"""Multi-GPU execution not supported right now. Using GPU
        {_A.gpu_ids[0]}"""
        )

    # --------------------------------------------------------------------------
    #   INSTANTIATE VOCABULARY, TOKENIZER, DATALOADER, MODEL, OPTIMIZER
    # --------------------------------------------------------------------------

    vocabulary = SentencePieceVocabulary(_C.DATA.VOCABULARY)
    tokenizer = SentencePieceTokenizer(_C.DATA.TOKENIZER)

    train_dataset = MaskedLanguageModelingDataset(
        lmdb_path=_C.DATA.TRAIN_LMDB,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
    )
    # TODO (kd): Use DistributedDataParalell on this ones.
    visual_module = VisualStream()
    linguistic_module = LinguisticStream.from_config(_C)
    model = ViswslModel(visual_module, linguistic_module).to(device)

    optimizer = OptimizerFactory.from_config(_C, model.parameters())
    lr_scheduler = LinearWarmupLinearDecayLR(
        optimizer,
        total_epochs=_C.OPTIM.NUM_ITERATIONS,
        warmup_proportion=_C.OPTIM.WARMUP_PROPORTION,
    )

    # --------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # --------------------------------------------------------------------------

    # Tensorboard summary writer for logging losses and metrics.
    tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)

    # Checkpoint manager to serialize checkpoints periodically while training
    # There is no notion of "best" checkpoint right now.
    checkpoint_manager = CheckpointManager(
        model, optimizer, _A.serialization_dir
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter: Iterator = iter(train_dataloader)

    # --------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------
    for iteration in tqdm(range(1, _C.OPTIM.NUM_ITERATIONS + 1)):

        # keys: {"image_id", "image", "caption_tokens", "masked_labels"}
        batch = next(train_dataloader_iter)
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        output_dict = model(
            batch["image"], batch["caption_tokens"], batch["masked_labels"]
        )
        batch_loss = output_dict["loss"].mean()
        batch_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), _C.OPTIM.CLIP_GRADIENTS)
        optimizer.step()
        lr_scheduler.step()

        # Log loss and learning rate to tensorboard.
        tensorboard_writer.add_scalar("loss", batch_loss, iteration)
        tensorboard_writer.add_scalar(
            "learning_rate", optimizer.param_groups[0]["lr"], iteration
        )

        # ----------------------------------------------------------------------
        #   PRINT EXAMPLES
        # ----------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            model.eval()

            # We will add qualitative examples directive to tensorboard.
            tensorboard_examples_text = ""

            for tokens, labels, predictions in zip(
                batch["caption_tokens"][:10],
                batch["masked_labels"][:10],
                output_dict["predictions"][:10],
            ):
                tokens = [
                    vocabulary.get_token_from_index(t.item()) for t in tokens
                ]
                labels = [
                    vocabulary.get_token_from_index(l.item()) for l in labels
                ]
                predictions = [
                    vocabulary.get_token_from_index(p.item())
                    for p in predictions
                ]

                tensorboard_examples_text += f"""
                    Caption tokens: {tokenizer.detokenize(tokens).replace("<unk>", "")}
                    Masked Labels: {" ".join(labels).replace("<unk>", "")}
                    Predictions: {" ".join(predictions).replace("<unk>", "")}

                    """
            tensorboard_writer.add_text(
                "qualitative", tensorboard_examples_text, iteration
            )
            checkpoint_manager.step(iteration)
            model.train()
