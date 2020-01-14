from typing import List

from torch.utils.data import IterableDataset

from viswsl.data.dataflows import (
    ReadDatapointsFromLmdb,
    TransformImageForResNetLikeModels,
    TokenizeCaption,
)

from viswsl.data.structures import CaptioningInstance, CaptioningBatch
from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary


class CaptioningDataset(IterableDataset):
    def __init__(
        self,
        lmdb_path: str,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        normalize_image: bool = False,
        image_resize_size: int = 256,
        image_crop_size: int = 224,
        max_caption_length: int = 30,
        shuffle: bool = False,
    ):
        # keys: {"image_id", "image", "captions"}
        self._pipeline = ReadDatapointsFromLmdb(lmdb_path, shuffle=shuffle)
        self._pipeline = TransformImageForResNetLikeModels(
            self._pipeline,
            normalize=normalize_image,
            image_resize_size=image_resize_size,
            image_crop_size=image_crop_size,
            index_or_key="image"
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
        self.max_caption_length = max_caption_length
        self.padding_idx = vocabulary.pad_index

    def __len__(self):
        return len(self._pipeline)

    def __iter__(self):
        self._pipeline.reset_state()

        for datapoint in self._pipeline:
            yield CaptioningInstance(
                datapoint["image_id"],
                datapoint["image"],
                datapoint["caption_tokens"],
            )

    def collate_fn(self, instances: List[CaptioningInstance]) -> CaptioningBatch:
        return CaptioningBatch(instances, padding_value=self.padding_idx)
