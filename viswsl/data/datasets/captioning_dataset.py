import random
from typing import Callable, List

import albumentations as alb
import numpy as np
from torch.utils.data import Dataset

from viswsl.data.readers import LmdbReader
from viswsl.data.structures import CaptioningInstance, CaptioningBatch
from viswsl.data.tokenizer import SentencePieceBPETokenizer
from viswsl.data.transforms import (
    IMAGENET_COLOR_MEAN,
    IMAGENET_COLOR_STD,
    NormalizeCaption,
    TokenizeCaption,
    TruncateCaptionTokens,
)


class CaptioningPretextDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        tokenizer: SentencePieceBPETokenizer,
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
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
    ):
        self.image_transform = image_transform
        self.reader = LmdbReader(lmdb_path, percentage=percentage)

        self.caption_transform = alb.Compose(
            [
                NormalizeCaption(),
                TokenizeCaption(tokenizer),
                TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.use_single_caption = use_single_caption
        self.padding_idx = tokenizer.token_to_id("<unk>")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> CaptioningInstance:

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
        return CaptioningInstance(image_id, image, caption_tokens)

    def collate_fn(self, instances: List[CaptioningInstance]) -> CaptioningBatch:
        return CaptioningBatch(instances, padding_value=self.padding_idx)
