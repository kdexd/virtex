import math
import random
from typing import Callable, List

import albumentations as alb
import numpy as np
import tokenizers as tkz
from torch.utils.data import IterableDataset

from viswsl.data.readers import LmdbReader
from viswsl.data.structures import WordMaskingInstance, WordMaskingBatch
from viswsl.data.transforms import (
    RandomHorizontalFlip,
    NormalizeCaption,
    TokenizeCaption,
    TruncateCaptionTokens,
)


class WordMaskingDataset(IterableDataset):

    def __init__(
        self,
        lmdb_path: str,
        tokenizer: tkz.implementations.BaseTokenizer,
        mask_proportion: float = 0.15,
        mask_probability: float = 0.80,
        replace_probability: float = 0.10,
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
        use_single_caption: bool = False,
        shuffle: bool = False,
    ):
        self._tokenizer = tokenizer
        self.image_transform = image_transform

        # keys: {"image_id", "image", "caption"}
        self._pipeline = LmdbReader(lmdb_path, shuffle=shuffle)

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

        # Handles to commonly used variables for word masking.
        self._mask_index = tokenizer.token_to_id("[MASK]")
        self._mask_proportion = mask_proportion
        self._mask_prob = mask_probability
        self._repl_prob = replace_probability

    def __iter__(self):
        self._pipeline.reset_state()

        for datapoint in self._pipeline:
            # Transform and convert image from HWC to CHW format.
            image = self.image_transform(image=datapoint["image"])["image"]
            image = np.transpose(image, (2, 0, 1))

            # Pick a random caption and process (transform) it.
            # Pick a random caption or first caption and process (transform) it.
            captions = datapoint["captions"]
            if self.use_single_caption:
                caption = captions[0]
            else:
                caption = random.choice(captions)

            caption_tokens = self.caption_transform(caption=caption)["caption"]

            # -----------------------------------------------------------------
            #  Mask some tokens randomly.
            # -----------------------------------------------------------------
            masked_labels = [self.padding_idx] * len(caption_tokens)

            # Indices in `caption_tokens` list to mask (minimum 1 index).
            # Leave out first and last indices (boundary tokens).
            tokens_to_mask: List[int] = random.sample(
                list(range(1, len(caption_tokens) - 1)),
                math.ceil((len(caption_tokens) - 2) * self._mask_proportion),
            )
            for i in tokens_to_mask:
                # Whether to replace with [MASK] or random word.
                # If only one token, always [MASK].
                if len(tokens_to_mask) == 1:
                    masked_labels[i] = caption_tokens[i]
                    caption_tokens[i] = self._mask_index
                else:
                    _flag: float = random.random()
                    if _flag <= self._mask_prob + self._repl_prob:
                        if _flag <= self._mask_prob:
                            masked_labels[i] = caption_tokens[i]
                            caption_tokens[i] = self._mask_index
                        else:
                            caption_tokens[i] = self._random_token_index()
            # -----------------------------------------------------------------

            yield WordMaskingInstance(
                datapoint["image_id"], image, caption_tokens, masked_labels
            )

    def collate_fn(self, instances: List[WordMaskingInstance]) -> WordMaskingBatch:
        return WordMaskingBatch(instances, padding_value=self.padding_idx)

    def _random_token_index(self) -> int:
        return random.randint(0, self._tokenizer.get_vocab_size() - 1)
