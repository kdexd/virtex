from collections import defaultdict
import glob
import json
import os
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

from virtex.data import transforms as T


class ImageNetDataset(ImageNet):
    r"""
    Simple wrapper over torchvision's ImageNet dataset with a feature to support
    restricting dataset size for semi-supervised learning setup (data-efficiency
    ablations).

    We also handle image transform here instead of passing to super class.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/imagenet")
        Path to the dataset root directory. This must contain directories
        ``train``, ``val`` with per-category sub-directories.
    split: str, optional (default = "train")
        Which split to read from. One of ``{"train", "val"}``.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    percentage: int, optional (default = 100)
        Percentage of dataset to keep. This dataset retains first K% of images
        per class to retain same class label distribution. This is 100% by
        default, and will be ignored if ``split`` is ``val``.
    """

    def __init__(
        self,
        data_root: str = "datasets/imagenet",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        percentage: float = 100,
    ):
        super().__init__(data_root, split)
        assert percentage > 0, "Cannot load dataset with 0 percent original size."

        self.image_transform = image_transform

        # Super class has `imgs` list and `targets` list. Make a dict of
        # class ID to index of instances in these lists and pick first K%.
        if split == "train" and percentage < 100:
            label_to_indices: Dict[int, List[int]] = defaultdict(list)
            for index, target in enumerate(self.targets):
                label_to_indices[target].append(index)

            # Trim list of indices per label.
            for label in label_to_indices:
                retain = int(len(label_to_indices[label]) * (percentage / 100))
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, label = super().__getitem__(idx)

        # Apply transformation to  image and convert to CHW format.
        image = self.image_transform(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1))
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "label": torch.stack([d["label"] for d in data], dim=0),
        }


class INaturalist2018Dataset(Dataset):
    r"""
    A dataset which provides image-label pairs from the iNaturalist 2018 dataset.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/inaturalist")
        Path to the dataset root directory. This must contain images and
        annotations (``train2018``, ``val2018`` and ``annotations`` directories).
    split: str, optional (default = "train")
        Which split to read from. One of ``{"train", "val"}``.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    """

    def __init__(
        self,
        data_root: str = "datasets/inaturalist",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        self.split = split
        self.image_transform = image_transform

        annotations = json.load(
            open(os.path.join(data_root, "annotations", f"{split}2018.json"))
        )
        # Make a list of image IDs to file paths.
        self.image_id_to_file_path = {
            ann["id"]: os.path.join(data_root, ann["file_name"])
            for ann in annotations["images"]
        }
        # For a list of instances: (image_id, category_id) tuples.
        self.instances = [
            (ann["image_id"], ann["category_id"])
            for ann in annotations["annotations"]
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_id, label = self.instances[idx]
        image_path = self.image_id_to_file_path[image_id]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "label": torch.stack([d["label"] for d in data], dim=0),
        }


class VOC07ClassificationDataset(Dataset):
    r"""
    A dataset which provides image-label pairs from the PASCAL VOC 2007 dataset.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/VOC2007")
        Path to the dataset root directory. This must contain directories
        ``Annotations``, ``ImageSets`` and ``JPEGImages``.
    split: str, optional (default = "trainval")
        Which split to read from. One of ``{"trainval", "test"}``.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    """

    def __init__(
        self,
        data_root: str = "datasets/VOC2007",
        split: str = "trainval",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        self.split = split
        self.image_transform = image_transform

        ann_paths = sorted(
            glob.glob(os.path.join(data_root, "ImageSets", "Main", f"*_{split}.txt"))
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
            (
                os.path.join(data_root, "JPEGImages", f"{image_name}.jpg"),
                label.tolist(),
            )
            for image_name, label in image_names_to_labels.items()
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_path, label = self.instances[idx]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "label": torch.stack([d["label"] for d in data], dim=0),
        }


class ImageDirectoryDataset(Dataset):
    r"""
    A dataset which reads images from any directory. This class is useful to
    run image captioning inference on our models with any arbitrary images.

    Parameters
    ----------
    data_root: str
        Path to a directory containing images.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    """

    def __init__(
        self, data_root: str, image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM
    ):
        self.image_paths = glob.glob(os.path.join(data_root, "*"))
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        # Remove extension from image name to use as image_id.
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        # Return image id as string so collate_fn does not cast to torch.tensor.
        return {"image_id": str(image_id), "image": torch.tensor(image)}
