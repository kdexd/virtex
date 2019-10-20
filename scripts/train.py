import argparse
import os
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
import viswsl.utils.distributed as dist
from viswsl.utils.logging import Timer


# fmt: off
parser = argparse.ArgumentParser(
    description="""Train only the linguistic stream (transformer) on masked
    language modeling pretext task."""
)
parser.add_argument(
    "--config", help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override", nargs="*",
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
    Setting this flag assumes ignores arguments `--num-gpus-per-machine`,
    `--num-machines`, `--machine-rank` and `--dist-url`. Set `$MASTER_PORT`
    env variable externally for distributed process group communication."""
)
parser.add_argument(
    "--num-gpus-per-machine", type=int, default=0,
    help="Number of GPUs per machine with IDs as 0, 1, 2.. and so on.",
)
parser.add_argument(
    "--num-machines", type=int, default=1,
    help="Number of machines used in distributed training."
)
parser.add_argument(
    "--machine-rank", type=int, default=0,
    help="""Rank of the machine, integer in [0, num_machines). Default 0 for
    training with a single machine.""",
)
parser.add_argument(
    "--dist-url", default=f"tcp://127.0.0.1:23456",
    help="""URL of the master process in distributed training, it defaults to
    localhost for single-machine training.""",
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

    device_id = dist.init_distributed_env(_A.dist_backend) if _A.slurm else 0
    device = torch.device("cuda", device_id)

    # -------------------------------------------------------------------------
    #   INSTANTIATE VOCABULARY, TOKENIZER, DATALOADER, MODEL, OPTIMIZER
    # -------------------------------------------------------------------------

    vocabulary = SentencePieceVocabulary(_C.DATA.VOCABULARY)
    tokenizer = SentencePieceTokenizer(_C.DATA.TOKENIZER)

    train_dataset = MaskedLanguageModelingDataset(
        lmdb_path=_C.DATA.TRAIN_LMDB,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        normalize_image=_C.DATA.NORMALIZE_IMAGE,
        max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE // dist.get_world_size(),
        num_workers=_A.cpu_workers,
    )

    visual_module = VisualStream()
    linguistic_module = LinguisticStream.from_config(_C)
    model = ViswslModel(visual_module, linguistic_module).to(device)

    if dist.get_world_size() > 1:
        dist.synchronize()
        model = nn.parallel.DistributedDataParallel(  # type: ignore
            model, device_ids=[device]
        )

    optimizer = OptimizerFactory.from_config(_C, model.parameters())
    lr_scheduler = LinearWarmupLinearDecayLR(
        optimizer,
        total_epochs=_C.OPTIM.NUM_ITERATIONS,
        warmup_proportion=_C.OPTIM.WARMUP_PROPORTION,
    )

    # -------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------

    # Only the master process would log and serialize checkpoints.
    if dist.is_master_process():
        tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)
        checkpoint_manager = CheckpointManager(
            model, optimizer, _A.serialization_dir
        )

    # Keep track of (moving) average time per iteration and ETA.
    timer = Timer(
        window_size=_A.log_every, total_iterations=_C.OPTIM.NUM_ITERATIONS
    )
    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter: Iterator = iter(train_dataloader)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------

    for iteration in range(1, _C.OPTIM.NUM_ITERATIONS + 1):
        timer.tic()
        # keys: {"image_id", "image", "caption_tokens", "masked_labels"}
        batch = next(train_dataloader_iter)
        for key in batch:
            batch[key] = batch[key].to(device)

        # keys; {"predictions", "loss"}
        output_dict = model(
            batch["image"], batch["caption_tokens"], batch["masked_labels"]
        )
        # Normalize the loss, because gradients are being accumulated (summed)
        # while the loss is averaged across training instances.
        loss = output_dict["loss"].mean() / _C.OPTIM.GRAD_ACCUMULATION_STEPS
        loss.backward()

        if iteration % _C.OPTIM.GRAD_ACCUMULATION_STEPS == 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), _C.OPTIM.CLIP_GRADIENTS
            )
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()
        timer.toc()

        # Make the master process log loss, lr, time to tensorboard.
        if iteration % _A.log_every == 0:
            loss = dist.average_across_processes(loss)
            if dist.is_master_process():
                # Print avg time and ETA for debugging: replacement of tqdm.
                print(timer.stats)

                tensorboard_writer.add_scalar("loss", loss, iteration)
                tensorboard_writer.add_scalar(
                    "learning_rate", optimizer.param_groups[0]["lr"], iteration
                )
                tensorboard_writer.add_scalar("avg_time", timer.avg, iteration)
                tensorboard_writer.add_scalar(
                    "eta_hours", timer.eta_sec / 3600, iteration
                )
            dist.synchronize()

        # ----------------------------------------------------------------------
        #   PRINT EXAMPLES
        # ----------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            # Remove all accumulated gradients before evaluation.
            optimizer.zero_grad()

            model.eval()

            # Make the master process log examples to tensorboard, and save
            # the model checkpoint.
            # This condition should always be True without distributed training.
            if dist.is_master_process():
                with torch.no_grad():
                    # We will add qualitative examples directive to tensorboard.
                    tensorboard_examples_text = ""

                    # Each process would gather some examples.
                    examples_per_rank = max(10 // dist.get_world_size(), 1)
                    for tokens, labels, predictions in zip(
                        batch["caption_tokens"][:examples_per_rank],
                        batch["masked_labels"][:examples_per_rank],
                        output_dict["predictions"][:examples_per_rank],
                    ):
                        tokens = [
                            vocabulary.get_token_from_index(t.item())
                            for t in tokens
                            if t.item() != vocabulary.unk_index
                        ]
                        labels = [
                            vocabulary.get_token_from_index(l.item())
                            for l in labels
                            if l.item() != vocabulary.unk_index
                        ]
                        predictions = [
                            vocabulary.get_token_from_index(p.item())
                            for p in predictions
                            if p.item() != vocabulary.unk_index
                        ]
                        tensorboard_examples_text += f"""
                            Caption tokens: {tokenizer.detokenize(tokens)}
                            Masked Labels: {" ".join(labels)}
                            Predictions: {" ".join(predictions)}

                            """

                tensorboard_writer.add_text(
                    "qualitative", tensorboard_examples_text, iteration
                )
                checkpoint_manager.step(iteration)

            # All processes will wait till master process is done logging.
            dist.synchronize()
            model.train()
