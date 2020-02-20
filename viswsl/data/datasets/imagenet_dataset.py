from typing import Callable, List

import albumentations as alb
import numpy as np
from torchvision.datasets import ImageNet

from viswsl.data.dataflows.transforms import AlexNetPCA


class ImageNetDataset(ImageNet):
    r"""
    Simple wrapper over torchvision's super class, we handle image transform
    here instead of passing to constructor.
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__(root, split)

        # Build desired tranform for ImageNet linear classification protocol.
        # Resize smaller size to 256, crop image (random crop if training,
        # center crop if evaluation), and normalize image by ImageNet color
        # mean. This should ideally not be changed.
        Crop: Callable = alb.CenterCrop if split == "train" else alb.RandomCrop
        transform_list: List[Callable] = [
            alb.SmallestMaxSize(256, always_apply=True),
            Crop(224, 224, always_apply=True),
        ]
        # During training, we also do color jitter and lighting noise.
        photometric_transforms = [
            alb.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            alb.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5
            ),
            AlexNetPCA(p=0.5),
        ]
        if split == "train":
            transform_list.extend(photometric_transforms)

        transform_list.extend(
            [
                alb.ToFloat(max_value=255.0, always_apply=True),
                alb.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                    always_apply=True,
                ),
            ]
        )
        # Super class handle transformation, but let's do this in this class
        # because albumentations.Compose accepts inputs differently than
        # `torchvision.transforms.Compose`.
        self.image_transform = alb.Compose(transform_list)

    def __getitem__(self, idx: int):
        image, label = super().__getitem__(idx)

        # Apply transformation to  image and convert to CHW format.
        image = self.image_transform(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1))

        # Convert `tuple` to a `dict`, so the return value is uniform with
        # `VOC07ClassificationDataset`.
        return {"image": image, "label": label}
