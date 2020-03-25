from collections import defaultdict
import glob
import json
import os
from typing import Callable, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from viswsl.data.structures import CaptioningInstance, CaptioningBatch
from viswsl.data import transforms as T


class InstanceClassificationDataset(Dataset):
    def __init__(
        self,
        root: str = "datasets/coco",  # TODO (kd): remove default value.
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        split: str = "train",
    ):
        self.image_transform = image_transform
        image_filenames = glob.glob(os.path.join(root, f"{split}2017", "*.jpg"))

        # Make a tuple of image id and its filename, get image_id from its
        # filename (assuming directory has images with names in COCO 2017 format).
        self.id_filename: List[Tuple[int, str]] = [
            (int(os.path.basename(name)[:-4]), name) for name in image_filenames
        ]

        # Load the instance (bounding box and mask) annotations.
        _annotations = json.load(
            open(os.path.join(root, "annotations", f"instances_{split}2017.json"))
        )
        # Make a mapping between COCO category id and its index, to make IDs
        # consecutive, else COCO has 80 classes with IDs 1-90. Start index from 1
        # as 0 is reserved for background (padding idx).
        _category_ids = {
            ann["id"]: index + 1
            for index, ann in enumerate(_annotations["categories"])
        }

        # A mapping between image ID and list of unique category IDs (indices as above)
        # in corresponding image.
        self.instances = defaultdict(list)

        for ann in _annotations["annotations"]:
            self.instances[ann["image_id"]].append(_category_ids[ann["category_id"]])

        # De-duplicate instances and drop empty labels, we only need to do
        # classification.
        self.instances = {
            image_id: list(set(ins))
            for image_id, ins in self.instances.items()
            if len(ins) > 0
        }
        # Filter out image IDs which didn't have any instances.
        self.id_filename = [
            (t[0], t[1]) for t in self.id_filename if t[0] in self.instances
        ]
        # Padding while forming a batch, because images may have variable number
        # of instances. We do not need padding index from tokenizer: COCO has
        # category ID 0 as background, conventionally.
        self.padding_idx = 0

    def __len__(self):
        return len(self.id_filename)

    def __getitem__(self, idx: int):
        # Get image ID and filename.
        image_id, filename = self.id_filename[idx]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        # Get a list of instances present in the image.
        instances = self.instances[image_id]

        # Treat list of instances as "caption tokens" for reusability.
        # TODO (kd): it is hacky and written in deadline rush, make it better.
        return CaptioningInstance(image_id, image, caption_tokens=instances)

    def collate_fn(self, instances: List[CaptioningInstance]) -> CaptioningBatch:
        return CaptioningBatch(instances, padding_value=self.padding_idx)
