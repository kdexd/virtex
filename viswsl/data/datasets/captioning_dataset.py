import random
from typing import Callable, List

import albumentations as alb
import numpy as np
import tokenizers as tkz
from torch.utils.data import Dataset

from viswsl.data.readers import LmdbReader
from viswsl.data.structures import CaptioningInstance, CaptioningBatch
from viswsl.data.transforms import (
    IMAGENET_COLOR_MEAN,
    IMAGENET_COLOR_STD,
    RandomHorizontalFlip,
    NormalizeCaption,
    TokenizeCaption,
    TruncateCaptionTokens,
)


class CaptioningPretextDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        tokenizer: tkz.implementations.BaseTokenizer,
        image_transform: Callable = alb.Compose(
            [
                alb.RandomResizedCrop(224, 224, always_apply=True),
                alb.ToFloat(always_apply=True),
                alb.Normalize(
                    mean=IMAGENET_COLOR_MEAN,
                    std=IMAGENET_COLOR_STD,
                    max_pixel_value=1.0,
                    always_apply=True,
                ),
            ]
        ),
        random_horizontal_flip: bool = True,
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
    ):
        self.image_transform = image_transform
        self.reader = LmdbReader(lmdb_path, percentage=percentage)

        # Random horizontal flip is kept separate from other data augmentation
        # transforms because we need to change the caption if image is flipped.
        if random_horizontal_flip:
            self.flip = RandomHorizontalFlip(p=0.5)
        else:
            # No-op is not required.
            self.flip = lambda x: x  # type: ignore

        self.caption_transform = alb.Compose(
            [
                NormalizeCaption(),
                TokenizeCaption(tokenizer),
                TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.use_single_caption = use_single_caption
        self.padding_idx = tokenizer.token_to_id("[UNK]")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> CaptioningInstance:

        image_id, image, captions = self.reader[idx]

        # Transform and convert image from HWC to CHW format.
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        # Pick a random caption or first caption and process (transform) it.
        if self.use_single_caption:
            caption = captions[0]
        else:
            caption = random.choice(captions)

        caption_tokens = self.caption_transform(caption=caption)["caption"]
        return CaptioningInstance(image_id, image, caption_tokens)

    def collate_fn(self, instances: List[CaptioningInstance]) -> CaptioningBatch:
        return CaptioningBatch(instances, padding_value=self.padding_idx)
