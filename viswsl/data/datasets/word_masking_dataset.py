import math
import random
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
from viswsl.data.structures import WordMaskingInstance, WordMaskingBatch


class WordMaskingDataset(IterableDataset):

    # TODO (kd) :document it later properly.
    # List of augmentations to be applied on each image after reading
    # from LMDB. This follows the standard augmentation steps of
    # (ImageNet pre-trained) ResNet models:
    #     1. Resize shortest edge to 256. (Already done in LMDB)
    #     2. Convert pixel intensities in [0, 1].
    #     3. Random crop a (224, 224) patch.
    #     4. Normalize image by mean ImageNet pixel intensity and
    #        variance (optional).
    #     5. Convert from HWC to CHW format.

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
        shuffle: bool = False,
    ):
        self._tokenizer = tokenizer
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

        # Handles to commonly used variables for word masking.
        self._mask_index = tokenizer.token_to_id("[MASK]")
        self._pad_index = tokenizer.token_to_id("[UNK]")
        self._mask_proportion = mask_proportion
        self._mask_prob = mask_probability
        self._repl_prob = replace_probability

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
        while True:
            token_index = random.randint(0, len(self._vocabulary) - 1)
            if token_index not in self._vocabulary.special_indices:
                return token_index
