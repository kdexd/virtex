import argparse
from collections import defaultdict
import os
import random
import sys
from typing import Dict, List

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
from viswsl.data.datasets import VOC07ClassificationDataset
from viswsl.factories import VisualStreamFactory, TextualStreamFactory
from viswsl.model import ViswslModel, VOC07ClassificationFeatureExtractor


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
# fmt: on


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()
    _C = Config(_A.config, _A.config_override)
    _C_DOWNSTREAM = _C.DOWNSTREAM.VOC07_CLF

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
    #   EXTRACT FEATURES FOR TRAINING SVMs
    # -------------------------------------------------------------------------

    train_dataset = VOC07ClassificationDataset(
        _C_DOWNSTREAM.DATA_ROOT, split="train"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=_C_DOWNSTREAM.BATCH_SIZE, pin_memory=True
    )
    test_dataset = VOC07ClassificationDataset(_C_DOWNSTREAM.DATA_ROOT, split="val")
    test_dataloader = DataLoader(
        test_dataset, batch_size=_C_DOWNSTREAM.BATCH_SIZE, pin_memory=True
    )
    NUM_CLASSES = len(train_dataset.class_names)

    # Initialize from a checkpoint, but only keep the visual module.
    model = ViswslModel(
        VisualStreamFactory.from_config(_C), TextualStreamFactory.from_config(_C)
    ).to(device)
    model.load_state_dict(torch.load(_A.checkpoint_path)["model"])

    feature_extractor = VOC07ClassificationFeatureExtractor(model, mode="avg")
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
            features = feature_extractor(
                batch["image"].to(device), _C_DOWNSTREAM.LAYER_NAMES
            )
            for layer_name in features:
                features_train[layer_name].append(
                    features[layer_name].detach().cpu()
                )

        # Similarly extract test features.
        for batch in tqdm(test_dataloader, desc="Extracting test features:"):
            targets_test.append(batch["label"])
            features = feature_extractor(
                batch["image"].to(device), _C_DOWNSTREAM.LAYER_NAMES
            )
            for layer_name in features:
                features_test[layer_name].append(features[layer_name].detach().cpu())

    # Convert batches of features/targets to one large numpy array
    features_train = {
        k: torch.cat(v, dim=0).numpy() for k, v in features_train.items()
    }
    features_test = {
        k: torch.cat(v, dim=0).numpy() for k, v in features_test.items()
    }
    targets_train = torch.cat(targets_train, dim=0).numpy()
    targets_test = torch.cat(targets_test, dim=0).numpy()

    # -------------------------------------------------------------------------
    #   TRAIN SVMs WITH EXTRACTED FEATURES
    # -------------------------------------------------------------------------

    # Store test set AP for each class, for features from every layer.
    # shape: (num_layers, num_classes)
    test_ap = np.zeros((len(_C_DOWNSTREAM.LAYER_NAMES), NUM_CLASSES))

    # Possible keys: {"layer1", "layer2", "layer3", "layer4"}
    for layer_idx, layer_name in enumerate(_C_DOWNSTREAM.LAYER_NAMES):

        # Iterate over all VOC classes and train one-vs-all linear SVMs.
        for cls_idx in range(NUM_CLASSES):
            cls_labels = targets_train[:, cls_idx].astype(dtype=np.int32, copy=True)
            # meaning of labels in VOC/COCO original loaded target files:
            # label 0 = not present, set it to -1 as svm train target
            # label 1 = present. Make the svm train target labels as -1, 1.
            cls_labels[np.where(cls_labels == 0)] = -1
            num_positives = len(np.where(cls_labels == 1)[0])
            num_negatives = len(cls_labels) - num_positives
            logger.info(
                f"""Class {train_dataset.class_names[cls_idx]}:
                    Positive Examples: {num_positives}
                    Negative Examples: {num_negatives}
                    Ratio: {num_positives / num_negatives}"""
            )

            # See which cost maximizes the AP for this class.
            max_crossval_ap: float = 0.0
            best_crossval_clf = None

            for cost in _C_DOWNSTREAM.SVM_COSTS:
                clf = LinearSVC(
                    C=cost,
                    class_weight={1: 2, -1: 1},
                    penalty="l2",
                    loss="squared_hinge",
                    max_iter=2000,
                )
                ap_scores = cross_val_score(
                    clf,
                    features_train[layer_name],
                    cls_labels,
                    cv=3,
                    scoring="average_precision",
                )
                clf.fit(features_train[layer_name], cls_labels)

                # Keep track of best SVM (based on cost) for each (layer, cls).
                if ap_scores.mean() > max_crossval_ap:
                    max_ap = ap_scores.mean()
                    best_crossval_clf = clf

                logger.info(
                    f"SVM for: {layer_name}, cost {cost}, mAP {ap_scores.mean()}"
                )

            # -----------------------------------------------------------------
            #   TEST THE TRAINED SVM (PER LAYER, PER CLASS)
            # -----------------------------------------------------------------

            predictions = best_crossval_clf.decision_function(
                features_test[layer_name]
            )
            # Meaning of labels in VOC/COCO original loaded target files:
            # label 0 = not present, set it to -1 as SVM train target.
            # label 1 = present. Make the SVM train target labels as -1, 1.
            cls_labels = targets_test[:, cls_idx].astype(dtype=np.int32, copy=True)
            evaluate_data_inds = targets_test[:, cls_idx] != -1
            eval_preds = predictions[evaluate_data_inds]
            eval_cls_labels = cls_labels[evaluate_data_inds]
            eval_cls_labels[np.where(eval_cls_labels == 0)] = -1

            # Binarize class labels to make AP targets.
            targets = eval_cls_labels > 0
            test_ap[layer_idx][cls_idx] = average_precision_score(
                targets, eval_preds
            )

        layer_test_map = np.mean(test_ap, axis=-1)[layer_idx]
        logger.info(f"mAP for {layer_name}: {layer_test_map}")
        tensorboard_writer.add_scalars(
            "metrics/voc07_clf",
            {f"{layer_name}_mAP": layer_test_map},
            CHECKPOINT_ITERATION,
        )

    best_test_map = max(np.mean(test_ap, axis=-1))
    tensorboard_writer.add_scalars(
        "metrics/voc07_clf", {"best_mAP": best_test_map}, CHECKPOINT_ITERATION
    )
