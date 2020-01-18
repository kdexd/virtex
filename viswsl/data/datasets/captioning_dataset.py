from typing import Callable, List

import albumentations as alb
import numpy as np
import tokenizers as tkz
from torch.utils.data import IterableDataset

from viswsl.data.dataflows import (
    ReadDatapointsFromLmdb,
    RandomHorizontalFlip,
    TokenizeCaption,
)
from viswsl.data.structures import CaptioningInstance, CaptioningBatch


class CaptioningDataset(IterableDataset):
    def __init__(
        self,
        lmdb_path: str,
        tokenizer: tkz.implementations.BaseTokenizer,
        image_transform: Callable = alb.Compose(
            [
                alb.SmallestMaxSize(max_size=256),
                alb.RandomCrop(224, 224),
                alb.ToFloat(max_value=255.0),
                alb.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
            ]
        ),
        random_horizontal_flip: bool = True,
        max_caption_length: int = 30,
        shuffle: bool = False,
    ):
        self.image_transform = image_transform
        # keys: {"image_id", "image", "caption"}
        self._pipeline = ReadDatapointsFromLmdb(lmdb_path, shuffle=shuffle)

        # Random horizontal flip is kept separate from other data augmentation
        # transforms because we need to change the caption if image is flipped.
        if random_horizontal_flip:
            self._pipeline = RandomHorizontalFlip(self._pipeline)

        # keys added: {"caption_tokens"}
        self._pipeline = TokenizeCaption(
            self._pipeline,
            tokenizer,
            input_key="caption",
            output_key="caption_tokens",
        )
        self.max_caption_length = max_caption_length
        self.padding_idx = tokenizer.token_to_id("[UNK]")

    def __len__(self):
        return len(self._pipeline)

    def __iter__(self):
        self._pipeline.reset_state()

        for datapoint in self._pipeline:
            # Transform and convert image from HWC to CHW format.
            image = self.image_transform(image=datapoint["image"])["image"]
            image = np.transpose(image, (2, 0, 1))

            # Trim captions up to maximum length.
            caption_tokens = datapoint["caption_tokens"]
            caption_tokens = caption_tokens[: self.max_caption_length]

            yield CaptioningInstance(datapoint["image_id"], image, caption_tokens)

    def collate_fn(self, instances: List[CaptioningInstance]) -> CaptioningBatch:
        return CaptioningBatch(instances, padding_value=self.padding_idx)
