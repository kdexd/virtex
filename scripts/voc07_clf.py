import argparse
from collections import defaultdict
import multiprocessing as mp
import os
from typing import Any, Dict, List

from loguru import logger
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from viswsl.config import Config
from viswsl.data import VOC07ClassificationDataset
from viswsl.factories import PretrainingModelFactory
from viswsl.models import FeatureExtractor9k
from viswsl.utils.checkpointing import CheckpointManager
from viswsl.utils.common import common_setup


# fmt: off
parser = argparse.ArgumentParser(
    description="""Train SVMs on intermediate features of pre-trained
    ResNet-like models for Pascal VOC2007 classification."""
)
parser.add_argument(
    "--config", required=True,
    help="""Path to a config file used to train the model whose checkpoint will
    be loaded."""
)
parser.add_argument(
    "--config-override", nargs="*", default=[],
    help="""A sequence of key-value pairs specifying certain config arguments
    (with dict-like nesting) using a dot operator.""",
)
parser.add_argument(
    "--gpu-id", type=int, default=0, help="ID of GPU to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=2, help="Number of CPU workers."
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
    "--checkpoint-path",
    help="""Path to load checkpoint and run downstream task evaluation. The
    name of checkpoint file is required to be `checkpoint_*.pth`, where * is
    iteration number from which the checkpoint was serialized."""
)
parser.add_argument(
    "--serialization-dir", default="/tmp/voc07_clf",
    help="""Path to a directory to save results log as a Tensorboard event
    file. Recommended to set as parent directory of checkpoint path for
    `weight_init = checkpoint`."""
)
# fmt: on


def train_test_single_svm(args):

    feats_train, tgts_train, feats_test, tgts_test, layer_name, cls_name, costs = (
        args
    )

    cls_labels = np.copy(tgts_train)
    # Meaning of labels in VOC/COCO original loaded target files:
    # label 0 = not present, set it to -1 as svm train target
    # label 1 = present. Make the svm train target labels as -1, 1.
    cls_labels[np.where(cls_labels == 0)] = -1

    # See which cost maximizes the AP for this class.
    best_crossval_ap: float = 0.0
    best_crossval_clf = None
    best_cost: float = 0.0

    # fmt: off
    for cost in costs:
        clf = LinearSVC(
            C=cost, class_weight={1: 2, -1: 1}, penalty="l2",
            loss="squared_hinge", max_iter=2000,
        )
        ap_scores = cross_val_score(
            clf, feats_train, cls_labels, cv=3, scoring="average_precision",
        )
        clf.fit(feats_train, cls_labels)

        # Keep track of best SVM (based on cost) for each (layer, cls).
        if ap_scores.mean() > best_crossval_ap:
            best_crossval_ap = ap_scores.mean()
            best_crossval_clf = clf
            best_cost = cost

    logger.info(
        f"Best SVM for: {layer_name}, {cls_name}, cost {best_cost}, mAP {best_crossval_ap}"
    )
    # fmt: on

    # -------------------------------------------------------------------------
    #   TEST THE TRAINED SVM (PER LAYER, PER CLASS)
    # -------------------------------------------------------------------------
    predictions = best_crossval_clf.decision_function(feats_test)
    evaluate_data_inds = tgts_test != -1
    eval_preds = predictions[evaluate_data_inds]

    cls_labels = np.copy(tgts_test)
    eval_cls_labels = cls_labels[evaluate_data_inds]
    eval_cls_labels[np.where(eval_cls_labels == 0)] = -1

    # Binarize class labels to make AP targets.
    targets = eval_cls_labels > 0
    return average_precision_score(targets, eval_preds)


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()
    if _A.weight_init == "imagenet":
        _A.config_override.extend(["MODEL.VISUAL.PRETRAINED", True])

    _C = Config(_A.config, _A.config_override)
    _DOWNC = _C.DOWNSTREAM.VOC07_CLF

    device = torch.device(f"cuda:{_A.gpu_id}" if _A.gpu_id != -1 else "cpu")
    common_setup(_C, _A)

    # -------------------------------------------------------------------------
    #   INSTANTIATE MODEL AND LOGGING
    # -------------------------------------------------------------------------

    # Initialize from a checkpoint, but only keep the visual module.
    model = PretrainingModelFactory.from_config(_C)

    # Load weights according to the init method, do nothing for `random`, and
    # `imagenet` is already taken care of.
    if _A.weight_init == "checkpoint":
        ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)
    elif _A.weight_init == "torchvision":
        # Keep strict=False because this state dict may have weights for
        # last fc layer.
        model.visual.cnn.load_state_dict(
            torch.load(_A.checkpoint_path, map_location="cpu")["state_dict"],
            strict=False,
        )

    # -------------------------------------------------------------------------
    #   EXTRACT FEATURES FOR TRAINING SVMs
    # -------------------------------------------------------------------------

    train_dataset = VOC07ClassificationDataset(_DOWNC.DATA_ROOT, split="trainval")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_DOWNC.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    test_dataset = VOC07ClassificationDataset(_DOWNC.DATA_ROOT, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=_DOWNC.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    NUM_CLASSES = len(train_dataset.class_names)

    feature_extractor = FeatureExtractor9k(model, _DOWNC.LAYER_NAMES).to(device)
    del model

    # Possible keys: {"layer1", "layer2", "layer3", "layer4"}
    # Each key holds a list of numpy arrays, one per example.
    features_train: Dict[str, List[torch.Tensor]] = defaultdict(list)
    features_test: Dict[str, List[torch.Tensor]] = defaultdict(list)

    targets_train: List[torch.Tensor] = []
    targets_test: List[torch.Tensor] = []

    # VOC07 is small, extract all features and keep them in memory.
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Extracting train features:"):
            targets_train.append(batch["label"])

            # Keep features from only those layers which will be used to
            # train SVMs.
            # keys: {"layer1", "layer2", "layer3", "layer4"}
            features = feature_extractor(batch["image"].to(device))
            for layer_name in features:
                features_train[layer_name].append(
                    features[layer_name].detach().cpu()
                )

        # Similarly extract test features.
        for batch in tqdm(test_dataloader, desc="Extracting test features:"):
            targets_test.append(batch["label"])

            features = feature_extractor(batch["image"].to(device))
            for layer_name in features:
                features_test[layer_name].append(features[layer_name].detach().cpu())

    # Convert batches of features/targets to one large numpy array
    features_train = {
        k: torch.cat(v, dim=0).numpy() for k, v in features_train.items()
    }
    features_test = {
        k: torch.cat(v, dim=0).numpy() for k, v in features_test.items()
    }
    targets_train = torch.cat(targets_train, dim=0).numpy().astype(np.int32)
    targets_test = torch.cat(targets_test, dim=0).numpy().astype(np.int32)

    # -------------------------------------------------------------------------
    #   TRAIN AND TEST SVMs WITH EXTRACTED FEATURES
    # -------------------------------------------------------------------------

    input_args: List[Any] = []
    # Possible keys: {"layer1", "layer2", "layer3", "layer4"}
    for layer_idx, layer_name in enumerate(_DOWNC.LAYER_NAMES):

        # Iterate over all VOC classes and train one-vs-all linear SVMs.
        for cls_idx in range(NUM_CLASSES):
            # fmt: off
            input_args.append((
                features_train[layer_name], targets_train[:, cls_idx],
                features_test[layer_name], targets_test[:, cls_idx],
                layer_name, train_dataset.class_names[cls_idx], _DOWNC.SVM_COSTS,
            ))
            # fmt: on

    pool = mp.Pool(processes=_A.cpu_workers)
    pool_output = pool.map(train_test_single_svm, input_args)

    # -------------------------------------------------------------------------
    #   TENSORBOARD LOGGING (RELEVANT MAINLY FOR weight_init=checkpoint)
    # -------------------------------------------------------------------------

    # Tensorboard writer for logging mAP scores. This is useful especially
    # when weight_init=checkpoint (which maybe be coming from a training job).
    tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)

    # Test set AP for each class, for features from every layer.
    # shape: (num_layers, num_classes)
    test_ap = torch.tensor(pool_output).view(-1, NUM_CLASSES)

    if len(_DOWNC.LAYER_NAMES) > 1:
        for layer_idx, layer_name in enumerate(_DOWNC.LAYER_NAMES):
            layer_test_map = torch.mean(test_ap, dim=-1)[layer_idx]
            logger.info(f"mAP for {layer_name}: {layer_test_map}")

            # Tensorboard logging only when _A.weight_init == "checkpoint"
            if _A.weight_init == "checkpoint":
                tensorboard_writer.add_scalars(
                    "metrics/voc07_clf",
                    {f"{layer_name}_mAP": layer_test_map},
                    ITERATION,
                )

    best_test_map = torch.max(torch.mean(test_ap, dim=-1)).item()
    logger.info(f"Best mAP: {best_test_map}")

    # Tensorboard logging only when _A.weight_init == "checkpoint"
    if _A.weight_init == "checkpoint":
        tensorboard_writer.add_scalars(
            "metrics/voc07_clf", {"best_mAP": best_test_map}, ITERATION
        )
