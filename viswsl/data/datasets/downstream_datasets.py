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
    alb.SmallestMaxSize(256, always_apply=True),
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


def no_op_image_loader(image_path):
    r"""Needed to pass to super class of :class:`ImageNetDataset`."""
    return image_path


class ImageNetDataset(ImageNet):
    r"""
    Simple wrapper over torchvision's super class with two extra features.

    1. Support restricting dataset size for semi-supervised learning setup
       (data-efficiency ablations).
    2. Add option to cache the whole dataset (uint8 images) in memory during
       training.

    We also handle image transform here instead of passing to super class.

    Parameters
    ----------
    percentage: int, optional (default = 100)
        Percentage of dataset to keep. This dataset retains first K% of images
        per class to retain same class label distribution. This is 100% by
        default, and will be ignored if ``split`` is ``val``.
    cache_size: int, optional (default = -1)
        Cache these many images in memory during first epoch so we get some
        speedup from second epoch onwards. In case of distributed training,
        number of cached images will be this number per process. Turn off
        shuffling to avoid duplicate caching among processes.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        percentage: float = 100,
        cache_size: int = -1,
    ):
        super().__init__(root, split, loader=no_op_image_loader)
        # Pass a No-Op for loader so we get image path instead of a PIL image
        # in our `__getitem__`. This prevents unnecessary reads after caching.

        assert percentage > 0, "Cannot load dataset with 0 percent original size."

        if split == "train":
            transform_list = IMAGENET_AND_PLACES205_TRAIN_TRANSFORM_LIST
        else:
            transform_list = IMAGENET_AND_PLACES205_VAL_TRANSFORM_LIST

        # Super class handle transformation, but let's do this in this class
        # because albumentations. Compose accepts inputs differently than
        # `torchvision.transforms.Compose`.
        self.image_transform = alb.Compose(transform_list)

        # Super class has `imgs` list and `targets` list. Make a dict of
        # class ID to index of instances in these lists and pick first K%.
        if split == "train" and percentage < 100:
            label_to_indices: Dict[int, List[int]] = defaultdict(list)
            for index, target in enumerate(self.targets):
                label_to_indices[target].append(index)

            # Trim list of indices per label.
            for label in label_to_indices:
                retain = int((len(label_to_indices[label]) * percentage) // 100)
                label_to_indices[label] = label_to_indices[label][:retain]

            # Trim `self.imgs` and `self.targets` as per indices we have.
            retained_indices: List[int] = [
                index
                for indices_per_label in label_to_indices.values()
                for index in indices_per_label
            ]
            # Shorter dataset with size K% of original dataset, but almost same
            # class label distribution. super class will handle the rest.
            self.imgs = [self.imgs[i] for i in retained_indices]
            self.targets = [self.targets[i] for i in retained_indices]
            self.samples = self.imgs

        # Keep a cache of resized uint8 images (mapping from index to image).
        self.cache_size = cache_size
        self.cached_images: Dict[int, np.ndarray] = {}

        # Keep a smallest edge resize transform handy if we are caching images
        # so we coud resize images to (256 x 256) and fit them in memory.
        self.resize = alb.SmallestMaxSize(256, always_apply=True)

    def __getitem__(self, idx: int):
        image_path, label = super().__getitem__(idx)

        # Get image from cache, or open from image path and optionally cache it.
        if idx in self.cached_images:
            # Retrieve image from cache if it exists.
            image = self.cached_images[idx]
        elif len(self.cached_images) < self.cache_size:
            # If we have space in cache, add image ater we read.
            image = np.array(Image.open(image_path).convert("RGB"))
            image = self.resize(image=image)["image"]
            self.cached_images[idx] = image
        else:
            # If cache if full and index not in cache, just read the image.
            image = np.array(Image.open(image_path).convert("RGB"))

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
