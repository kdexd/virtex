from typing import Optional

import dataflow as df
import torch
from torch.utils.data import IterableDataset

from viswsl.data.dataflows import (
    ReadDatapointsFromLmdb,
    TransformImageForResNetLikeModels,
    TokenizeAndPadCaption,
    MaskSomeTokensRandomly,
)

from viswsl.config import Config
from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary


class MaskedLanguageModelingDataset(IterableDataset):
    def __init__(
        self,
        lmdb_path: str,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        mask_proportion: float = 0.15,
        mask_probability: float = 0.80,
        replace_probability: float = 0.10,
        normalize_image: bool = False,
        max_caption_length: int = 25,
        buffer_size: int = 64,
    ):
        assert buffer_size >= 0, "Buffer size must be non-negative."
        self._vocabulary = vocabulary

        self._pipeline = ReadDatapointsFromLmdb(lmdb_path)
        self._pipeline = TransformImageForResNetLikeModels(
            self._pipeline, normalize=normalize_image, index_or_key="image"
        )
        self._pipeline = TokenizeAndPadCaption(
            self._pipeline,
            vocabulary,
            tokenizer,
            max_caption_length=max_caption_length,
        )
        self._pipeline = MaskSomeTokensRandomly(
            self._pipeline,
            vocabulary=vocabulary,
            mask_proportion=mask_proportion,
            mask_probability=mask_probability,
            replace_probability=replace_probability,
        )
        # Keep a fixed-size buffer: examples will be pushed in this buffer and
        # randomly selected to make batches; a good proxy for random reads.
        # Set buffer size=0 to avoid shuffling and have finite length.
        if buffer_size > 0:
            self._pipeline = df.LocallyShuffleData(self._pipeline, buffer_size)

    @classmethod
    def from_config(
        cls,
        config: Config,
        vocabulary: Optional[SentencePieceVocabulary] = None,
        tokenizer: Optional[SentencePieceTokenizer] = None,
        split: str = "train",  # one of {"train", "val"}
    ):
        _C = config
        vocabulary = vocabulary or SentencePieceVocabulary(_C.DATA.VOCABULARY)
        tokenizer = tokenizer or SentencePieceTokenizer(_C.DATA.TOKENIZER)

        return cls(
            lmdb_path=_C.DATA.VAL_LMDB if split == "val" else _C.DATA.TRAIN_LMDB,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            mask_proportion=_C.PRETEXT.MASKED_LM.MASK_PROPORTION,
            mask_probability=_C.PRETEXT.MASKED_LM.MASK_PROBABILITY,
            replace_probability=_C.PRETEXT.MASKED_LM.REPLACE_PROBABILITY,
            normalize_image=_C.DATA.NORMALIZE_IMAGE,
            max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
            buffer_size=0 if split == "val" else 64,
        )

    def __len__(self):
        return len(self._pipeline)

    def __iter__(self):
        self._pipeline.reset_state()

        for datapoint in self._pipeline:
            # Mask out few tokens randomly.
            image_id = torch.tensor(datapoint["image_id"]).long()
            image = torch.tensor(datapoint["image"]).float()
            caption_tokens = torch.tensor(datapoint["caption_tokens"]).long()
            masked_labels = torch.tensor(datapoint["masked_labels"]).long()

            yield {
                "image_id": image_id,
                "image": image,
                "caption_tokens": caption_tokens,
                "masked_labels": masked_labels,
            }
