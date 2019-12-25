from typing import Optional

import torch
from torch.utils.data import IterableDataset

import dataflow as df
from viswsl.data.dataflows import (
    ReadDatapointsFromLmdb,
    TransformImageForResNetLikeModels,
    TokenizeCaption,
    MaskSomeTokensRandomly,
    PadSequences,
)

from viswsl.config import Config
from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary


class ImageCaptionDataset(IterableDataset):
    def __init__(
        self,
        lmdb_path: str,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        do_word_masking: bool = True,
        mask_proportion: float = 0.15,
        mask_probability: float = 0.80,
        replace_probability: float = 0.10,
        normalize_image: bool = False,
        max_caption_length: int = 25,
        shuffle: bool = False,
    ):
        self._vocabulary = vocabulary
        self._do_word_masking = do_word_masking

        # keys: {"image_id", "image", "captions"}
        self._pipeline = ReadDatapointsFromLmdb(lmdb_path, shuffle=shuffle)
        self._pipeline = TransformImageForResNetLikeModels(
            self._pipeline, normalize=normalize_image, index_or_key="image"
        )
        # keys added: {"caption_tokens"}
        self._pipeline = TokenizeCaption(
            self._pipeline,
            vocabulary,
            tokenizer,
            max_caption_length=max_caption_length,
            input_key="captions",
            output_key="caption_tokens",
        )
        if self._do_word_masking:
            # Create a copy of original caption, keys_added: {"masked_tokens"}
            self._pipeline = df.MapData(
                self._pipeline,
                lambda dp: dp.update(masked_tokens=dp["caption_tokens"]) or dp,
            )
            # keys added: {"masked_labels"}
            self._pipeline = MaskSomeTokensRandomly(
                self._pipeline,
                vocabulary=vocabulary,
                mask_proportion=mask_proportion,
                mask_probability=mask_probability,
                replace_probability=replace_probability,
                input_key="masked_tokens",
                output_key="masked_labels",
            )
            self._pipeline = PadSequences(
                self._pipeline,
                max_length=max_caption_length,
                padding_value=vocabulary.pad_index,
                input_key=["masked_tokens", "masked_labels"],
            )
        self._pipeline = PadSequences(
            self._pipeline,
            max_length=max_caption_length,
            padding_value=vocabulary.pad_index,
            input_key="caption_tokens",
        )

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
            do_word_masking=_C.MODEL.NAME == "word_masking",
            mask_proportion=_C.PRETEXT.WORD_MASKING.MASK_PROPORTION,
            mask_probability=_C.PRETEXT.WORD_MASKING.MASK_PROBABILITY,
            replace_probability=_C.PRETEXT.WORD_MASKING.REPLACE_PROBABILITY,
            normalize_image=_C.DATA.NORMALIZE_IMAGE,
            max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
            shuffle=False if split == "val" else True,
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

            instance = {
                "image_id": image_id,
                "image": image,
                "caption_tokens": caption_tokens,
            }
            if self._do_word_masking:
                masked_tokens = torch.tensor(datapoint["masked_tokens"]).long()
                masked_labels = torch.tensor(datapoint["masked_labels"]).long()
                instance.update(
                    masked_tokens=masked_tokens, masked_labels=masked_labels
                )
            yield instance
