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
    Simple wrapper over torchvision's ImageNet dataset. Image transform is
    handled here instead of passing to super class.

    Args:
        data_root: Path to the ImageNet dataset directory.
        split: Which split to read from. One of ``{"train", "val"}``.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
    """

    def __init__(
        self,
        data_root: str = "datasets/imagenet",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        super().__init__(data_root, split)
        self.image_transform = image_transform

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

    Args:
        data_root: Path to the iNaturalist 2018 dataset directory.
        split: Which split to read from. One of ``{"train", "val"}``.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
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

    Args:
        data_root: Path to VOC 2007 directory containing sub-directories named
            ``Annotations``, ``ImageSets``, and ``JPEGImages``.
        split: Which split to read from. One of ``{"trainval", "test"}``.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
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

    Args:
        data_root: Path to a directory containing images.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
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
