from collections import defaultdict
import glob
import json
import os
import random
from typing import Any, Callable, Dict, List, Tuple

import albumentations as alb
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from virtex.data.readers import LmdbReader
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T


class TokenClassificationDataset(Dataset):
    r"""
    A dataset which provides image-labelset pairs from a serialized LMDB file
    (COCO Captions in this codebase). the set of caption tokens (unordered)
    is treated as a labelset. Used for token classification pretraining task.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/coco")
        Path to the dataset root directory. This must contain the serialized
        LMDB files (for COCO ``train2017`` and ``val2017`` splits).
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    tokenizer: virtex.data.tokenizers.SentencePieceBPETokenizer
        A tokenizer which has the mapping between word tokens and their
        integer IDs.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    max_caption_length: int, optional (default = 30)
        Maximum number of tokens to keep in output caption tokens. Extra tokens
        will be trimmed from the right end of the token list.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
    ):
        lmdb_path = os.path.join(data_root, f"serialized_{split}.lmdb")
        self.reader = LmdbReader(lmdb_path)

        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.padding_idx = tokenizer.token_to_id("<unk>")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        image_id, image, captions = self.reader[idx]

        # Pick a random caption and then transform it.
        caption = random.choice(captions)

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))

        caption_tokens = self.caption_transform(caption=caption)["caption"]
        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(caption_tokens, dtype=torch.long),
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        labels = torch.nn.utils.rnn.pad_sequence(
            [d["labels"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        return {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "labels": labels,
        }


class MultiLabelClassificationDataset(Dataset):
    r"""
    A dataset which provides image-labelset pairs from COCO instance annotation
    files. This is used for multilabel classification pretraining task.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/coco")
        Path to the dataset root directory. This must contain images and
        annotations (``train2017``, ``val2017`` and ``annotations`` directories).
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        self.image_transform = image_transform

        # Make a tuple of image id and its filename, get image_id from its
        # filename (assuming directory has images with names in COCO 2017 format).
        image_filenames = glob.glob(os.path.join(data_root, f"{split}2017", "*.jpg"))
        self.id_filename: List[Tuple[int, str]] = [
            (int(os.path.basename(name)[:-4]), name) for name in image_filenames
        ]
        # Load the instance (bounding box and mask) annotations.
        _annotations = json.load(
            open(os.path.join(data_root, "annotations", f"instances_{split}2017.json"))
        )
        # Make a mapping between COCO category id and its index, to make IDs
        # consecutive, else COCO has 80 classes with IDs 1-90. Start index from 1
        # as 0 is reserved for background (padding idx).
        _category_ids = {
            ann["id"]: index + 1 for index, ann in enumerate(_annotations["categories"])
        }
        # Mapping from image ID to list of unique category IDs (indices as above)
        # in corresponding image.
        self._labels: Dict[str, Any] = defaultdict(list)

        for ann in _annotations["annotations"]:
            self._labels[ann["image_id"]].append(_category_ids[ann["category_id"]])

        # De-duplicate and drop empty labels, we only need to do classification.
        self._labels = {
            _id: list(set(lbl)) for _id, lbl in self._labels.items() if len(lbl) > 0
        }
        # Filter out image IDs which didn't have any labels.
        self.id_filename = [
            (t[0], t[1]) for t in self.id_filename if t[0] in self._labels
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
        labels = self._labels[image_id]

        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        labels = torch.nn.utils.rnn.pad_sequence(
            [d["labels"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        return {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "labels": labels,
        }
