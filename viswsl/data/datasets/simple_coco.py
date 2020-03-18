from collections import defaultdict
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Some simplified type renaming for better readability
ImageID = int
Captions = List[str]


class SimpleCocoCaptionsDataset(Dataset):
    def __init__(self, root: str = "datasets/coco", split: str = "train"):

        image_dir = os.path.join(root, f"{split}2017")

        # Make a tuple of image id and its filename, get image_id from its
        # filename (assuming directory has images with names in COCO2017 format).
        image_filenames = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.id_filename: List[Tuple[ImageID, str]] = [
            (int(os.path.basename(name)[:-4]), name) for name in image_filenames
        ]

        # Make a mapping between image_id and its captions.
        _captions = json.load(
            open(os.path.join(root, "annotations", f"captions_{split}2017.json"))
        )
        self._id_to_captions: Dict[ImageID, Captions] = defaultdict(list)

        for ann in _captions["annotations"]:
            self._id_to_captions[ann["image_id"]].append(ann["caption"])

    def __len__(self):
        return len(self.id_filename)

    def __getitem__(self, idx: int):
        image_id, filename = self.id_filename[idx]

        # shape: (height, width, channels), dtype: uint8
        image = np.array(Image.open(filename).convert("RGB"))
        captions = self._id_to_captions[image_id]

        return {"image_id": image_id, "image": image, "captions": captions}
