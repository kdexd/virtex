from typing import List

from torch.utils.data import IterableDataset

from viswsl.data.dataflows import (
    ReadDatapointsFromLmdb,
    TransformImageForResNetLikeModels,
    TokenizeCaption,
    MaskSomeTokensRandomly,
)

from viswsl.data.structures import WordMaskingInstance, WordMaskingBatch
from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary


class WordMaskingDataset(IterableDataset):
    def __init__(
        self,
        lmdb_path: str,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        mask_proportion: float = 0.15,
        mask_probability: float = 0.80,
        replace_probability: float = 0.10,
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
        # keys added: {"masked_labels"}
        self._pipeline = MaskSomeTokensRandomly(
            self._pipeline,
            vocabulary=vocabulary,
            mask_proportion=mask_proportion,
            mask_probability=mask_probability,
            replace_probability=replace_probability,
            input_key="caption_tokens",
            output_key="masked_labels",
        )
        self.max_caption_length = max_caption_length
        self.padding_idx = vocabulary.pad_index

    def __len__(self):
        return len(self._pipeline)

    def __iter__(self):
        self._pipeline.reset_state()

        for datapoint in self._pipeline:
            yield WordMaskingInstance(
                datapoint["image_id"],
                datapoint["image"],
                datapoint["caption_tokens"],
                datapoint["masked_labels"],
            )

    def collate_fn(self, instances: List[WordMaskingInstance]) -> WordMaskingBatch:
        return WordMaskingBatch(instances, padding_value=self.padding_idx)
