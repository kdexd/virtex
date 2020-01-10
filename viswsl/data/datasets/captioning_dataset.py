import torch
from torch.utils.data import IterableDataset

from viswsl.data.dataflows import (
    ReadDatapointsFromLmdb,
    TransformImageForResNetLikeModels,
    TokenizeCaption,
)

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
        self.max_caption_length = max_caption_length
        self.padding_idx = vocabulary.pad_index

    def __len__(self):
        return len(self._pipeline)

    def __iter__(self):
        self._pipeline.reset_state()

        for datapoint in self._pipeline:
            caption_tokens = datapoint["caption_tokens"]
            caption_length = len(caption_tokens)

            # Pad caption tokens to maximum length, so default collate_fn works.
            caption_tokens.extend(
                [self.padding_idx] * (self.max_caption_length - len(caption_tokens))
            )
            yield {
                "image_id": torch.tensor(datapoint["image_id"]).long(),
                "image": torch.tensor(datapoint["image"]).float(),
                "caption_tokens": torch.tensor(caption_tokens).long(),
                "caption_lengths": torch.tensor(caption_length).long(),
            }
