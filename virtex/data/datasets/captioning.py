import os
import random
from typing import Callable, List

import albumentations as alb
import numpy as np
from torch.utils.data import Dataset

from virtex.data.readers import LmdbReader
from virtex.data.structures import ImageCaptionInstance, ImageCaptionBatch
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T 


class CaptioningDataset(Dataset):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a serialized LMDB file (COCO Captions in this codebase). This is used for
    pretraining tasks which use captions - bicaptioning, forward captioning and
    token classification.

    This dataset also supports training on a randomly selected subset of the
    full dataset.

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
    use_single_caption: bool, optional (default = False)
        COCO Captions provides five captions per image. If this is True, only
        one fixed caption per image is use fo training (used for an ablation).
    percentage: float, optional (default = 100.0)
        Randomly sample this much percentage of full dataset for training.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
    ):
        lmdb_path = os.path.join(data_root, f"serialized_{split}.lmdb")
        self.reader = LmdbReader(lmdb_path, percentage=percentage)

        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.use_single_caption = use_single_caption
        self.padding_idx = tokenizer.token_to_id("<unk>")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> ImageCaptionInstance:

        image_id, image, captions = self.reader[idx]

        # Pick a random caption or first caption and process (transform) it.
        if self.use_single_caption:
            caption = captions[0]
        else:
            caption = random.choice(captions)

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))

        caption_tokens = self.caption_transform(caption=caption)["caption"]
        return ImageCaptionInstance(image_id, image, caption_tokens)

    def collate_fn(self, instances: List[ImageCaptionInstance]) -> ImageCaptionBatch:
        return ImageCaptionBatch(instances, padding_value=self.padding_idx)
