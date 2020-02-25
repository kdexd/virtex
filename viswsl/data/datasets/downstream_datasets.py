from collections import defaultdict
import csv
import glob
import os
from typing import Callable, Dict, List, Tuple

import albumentations as alb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

from viswsl.data.transforms import AlexNetPCA
from viswsl.data.structures import (
    LinearClassificationInstance,
    LinearClassificationBatch,
)


r"""
Desired image transform for ImageNet and Places205 linear classification
protocol. This follows evaluation protocol similar to several prior works::

    1. `(Misra et al, 2019) "Self-Supervised Learning of Pretext-Invariant
       Representations" <https://arxiv.org/abs/1912.01991>`_.

    2. `(Goyal et al, 2019) "Scaling and Benchmarking Self-Supervised Visual
       Representation Learning" <https://arxiv.org/abs/1905.01235>`_.

    3. `(Zhang et al, 2016a) "Colorful Image Colorization" 
       <https://arxiv.org/abs/1603.08511>`_.

    4. `(Zhang et al, 2016b) "Split-Brain Autoencoders: Unsupervised Learning
       by Cross-Channel Prediction" <https://arxiv.org/abs/1611.09842>`_.

This should ideally not be changed for apples-to-apples comparison.
"""
IMAGENET_AND_PLACES205_TRAIN_TRANSFORM_LIST = [
    alb.RandomResizedCrop(
        224, 224, scale=(0.08, 1.0), ratio=(0.75, 1.33), always_apply=True
    ),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    alb.HueSaturationValue(
        hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5
    ),
    AlexNetPCA(p=0.5),
    alb.ToFloat(max_value=255.0, always_apply=True),
    alb.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=1.0,
        always_apply=True,
    ),
]


r"""
Desired image transform for ImageNet and Places205 linear classification
protocol during validation phase. Consistent with prior works listed above.
"""
IMAGENET_AND_PLACES205_VAL_TRANSFORM_LIST = [
    alb.SmallestMaxSize(256, always_apply=True),
    alb.CenterCrop(224, 224, always_apply=True),
    alb.ToFloat(max_value=255.0, always_apply=True),
    alb.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=1.0,
        always_apply=True,
    ),
]


class ImageNetDataset(ImageNet):
    r"""
    Simple wrapper over torchvision's super class, we handle image transform
    here instead of passing to constructor.
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__(root, split)

        if split == "train":
            transform_list = IMAGENET_AND_PLACES205_TRAIN_TRANSFORM_LIST
        else:
            transform_list = IMAGENET_AND_PLACES205_VAL_TRANSFORM_LIST

        # Super class handle transformation, but let's do this in this class
        # because albumentations. Compose accepts inputs differently than
        # `torchvision.transforms.Compose`.
        self.image_transform = alb.Compose(transform_list)

    def __getitem__(self, idx: int):
        image, label = super().__getitem__(idx)

        # Apply transformation to  image and convert to CHW format.
        image = self.image_transform(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1))
        return LinearClassificationInstance(image=image, label=label)

    def collate_fn(
        self, instances: List[LinearClassificationInstance]
    ) -> LinearClassificationBatch:
        return LinearClassificationBatch(instances)


class Places205Dataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.split = split

        if split == "train":
            transform_list = IMAGENET_AND_PLACES205_TRAIN_TRANSFORM_LIST
        else:
            transform_list = IMAGENET_AND_PLACES205_VAL_TRANSFORM_LIST

        self.image_transform = alb.Compose(transform_list)

        # This directory contains all the images resized to (256 x 256).
        self._image_dir = os.path.join(
            root, "data", "vision", "torralba", "deeplearning", "images256"
        )
        # Path to annotatios CSV file corresponding to this split.
        annotations_path = os.path.join(
            root, "trainvalsplit_places205", f"{split}_places205.csv"
        )
        # Read annotation CSV file into tuples of (image_filename, label).
        self.instances: List[Tuple[str, int]] = []
        with open(annotations_path, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                self.instances.append((row[0], int(row[1])))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_filename, label = self.instances[idx]
        image_path = os.path.join(self._image_dir, image_filename)

        # Open image from path and apply transformation, convert to CHW format.
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        return LinearClassificationInstance(image=image, label=label)

    def collate_fn(
        self, instances: List[LinearClassificationInstance]
    ) -> LinearClassificationBatch:
        return LinearClassificationBatch(instances)


class VOC07ClassificationDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.split = split

        # ---------------------------------------------------------------------
        # Build desired tranform for VOC07 linear classification protocol.
        # This follows evaluation protocol similar to several prior works and
        # should ideally not be changed for apples-to-apples comparison.
        # This is DIFFERENT than ImageNet and Places205 transforms.
        # ---------------------------------------------------------------------
        self.image_transform: Callable = alb.Compose(
            [
                alb.Resize(224, 224, always_apply=True),
                alb.ToFloat(max_value=255.0, always_apply=True),
                alb.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                    always_apply=True,
                ),
            ]
        )
        # ---------------------------------------------------------------------

        ann_paths = sorted(
            glob.glob(os.path.join(root, "ImageSets", "Main", f"*_{split}.txt"))
        )
        # A list like; ["aeroplane", "bicycle", "bird", ...]
        self.class_names = [
            os.path.basename(path).split("_")[0] for path in ann_paths
        ]

        # We will construct a map for image name to a list of
        # shape: (num_classes, ) and values as one of {-1, 0, 1}.
        # 1: present, -1: not present, 0: ignore.
        image_names_to_labels: Dict[str, torch.Tensor] = defaultdict(
            lambda: -torch.ones(len(self.class_names), dtype=torch.int32)
        )
        for cls_num, ann_path in enumerate(ann_paths):
            with open(ann_path, "r") as fopen:
                for line in fopen:
                    img_name, orig_label_str = line.strip().split()
                    orig_label = int(orig_label_str)

                    # In VOC data, -1 (not present): set to 0 as train target
                    # In VOC data, 0 (ignore): set to -1 as train target.
                    orig_label = (
                        0 if orig_label == -1 else -1 if orig_label == 0 else 1
                    )
                    image_names_to_labels[img_name][cls_num] = orig_label

        # Convert the dict to a list of tuples for easy indexing.
        # Replace image name with full image path.
        self.instances: List[Tuple[str, torch.Tensor]] = [
            (os.path.join(root, "JPEGImages", f"{image_name}.jpg"), label.tolist())
            for image_name, label in image_names_to_labels.items()
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_path, label = self.instances[idx]

        # Open image from path and apply transformation, convert to CHW format.
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        return LinearClassificationInstance(image=image, label=label)  # type: ignore

    def collate_fn(
        self, instances: List[LinearClassificationInstance]
    ) -> LinearClassificationBatch:
        return LinearClassificationBatch(instances)
