import math
import os
import random
from typing import Callable, List

import albumentations as alb
import numpy as np
from torch.utils.data import Dataset

from virtex.data.readers import LmdbReader
from virtex.data.structures import WordMaskingInstance, WordMaskingBatch
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T


class WordMaskingPretextDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        mask_proportion: float = 0.15,
        mask_probability: float = 0.80,
        replace_probability: float = 0.10,
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

        # Handles to commonly used variables for word masking.
        self._vocab_size = tokenizer.get_vocab_size()
        self._mask_index = tokenizer.token_to_id("[MASK]")
        self._mask_proportion = mask_proportion
        self._mask_prob = mask_probability
        self._repl_prob = replace_probability

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> WordMaskingInstance:

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

        # ---------------------------------------------------------------------
        #  Mask some tokens randomly.
        # ---------------------------------------------------------------------
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
        # ---------------------------------------------------------------------

        return WordMaskingInstance(image_id, image, caption_tokens, masked_labels)

    def collate_fn(self, instances: List[WordMaskingInstance]) -> WordMaskingBatch:
        return WordMaskingBatch(instances, padding_value=self.padding_idx)

    def _random_token_index(self) -> int:
        return random.randint(0, self._vocab_size - 1)
