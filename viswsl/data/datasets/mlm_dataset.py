import dataflow as df
import torch
from torch.utils.data import IterableDataset

from viswsl.data.dataflows import (
    ReadDatapointsFromLmdb,
    TransformImageForResNetLikeModels,
    TokenizeAndPadCaption,
)

from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.utils.pretraining import mask_some_tokens_randomly


class MaskedLanguageModelingDataset(IterableDataset):
    def __init__(
        self,
        lmdb_path: str,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        normalize_image: bool = False,
        max_caption_length: int = 25,
        buffer_size: int = 64,
    ):
        assert buffer_size > 0, "Buffer size must be non-negative."
        self._vocabulary = vocabulary

        self._pipeline = ReadDatapointsFromLmdb(lmdb_path)
        self._pipeline = TransformImageForResNetLikeModels(
            self._pipeline, normalize=normalize_image
        )
        self._pipeline = TokenizeAndPadCaption(
            self._pipeline,
            vocabulary,
            tokenizer,
            max_caption_length=max_caption_length,
        )
        # Keep a fixed-size buffer: examples will be pushed in this buffer and
        # randomly selected to make batches; a good proxy for random reads.
        self._pipeline = df.LocallyShuffleData(self._pipeline, buffer_size)

    def __len__(self):
        return len(self._pipeline)

    def __iter__(self):
        self._pipeline.reset_state()

        for datapoint in self._pipeline:
            # Mask out few tokens randomly.
            caption_tokens, masked_labels = mask_some_tokens_randomly(
                datapoint["caption_tokens"],
                mask_token=self._vocabulary.mask_index,
                pad_token=self._vocabulary.pad_index,
                ignore_tokens=[
                    self._vocabulary.cls_index,
                    self._vocabulary.sep_index,
                ],
            )

            image_id = torch.tensor(datapoint["image_id"]).long()
            image = torch.tensor(datapoint["image"]).float()
            caption_tokens = torch.tensor(datapoint["caption_tokens"]).long()
            masked_labels = torch.tensor(masked_labels).long()

            yield {
                "image_id": image_id,
                "image": image,
                "caption_tokens": caption_tokens,
                "masked_labels": masked_labels,
            }
