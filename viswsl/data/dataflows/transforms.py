import random
from typing import Iterator, Union

import dataflow as df
import numpy as np

from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.types import LmdbDatapoint


class TransformImageForResNetLikeModels(df.ProxyDataFlow):

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
        ds: df.DataFlow,
        normalize: bool = False,
        index_or_key: Union[int, str] = "image",
    ):
        self.ds = ds
        self._normalize = normalize
        self._x = index_or_key

        # fmt: off
        self._augmentor = df.imgaug.AugmentorList([
            df.imgaug.RandomCrop(224),
            df.imgaug.ToFloat32(),
            df.imgaug.MapImage(self._transform),
        ])
        # fmt: on

    def _transform(self, image: np.ndarray) -> np.ndarray:
        image = image / 255.0
        if self._normalize:
            image -= np.array([0.485, 0.456, 0.406])
            image /= np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))
        return image

    def __iter__(self) -> Iterator[LmdbDatapoint]:

        for datapoint in self.ds:
            image = self._augmentor.augment(datapoint[self._x])
            datapoint[self._x] = image

            yield datapoint


class TokenizeAndPadCaption(df.DataFlow):
    def __init__(
        self,
        ds: df.DataFlow,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        max_caption_length: int = 25,
        index_or_key: Union[int, str] = "captions",
    ):
        self.ds = ds
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._x = index_or_key
        self._max_caption_length = max_caption_length

    def __iter__(self) -> Iterator[LmdbDatapoint]:

        for datapoint in self.ds:

            # Select a random caption, usually there may be only one caption.
            caption = random.choice(datapoint[self._x])

            # Tokenize caption (these are still strings).
            caption_tokens = self._tokenizer.tokenize(caption)

            # Add [CLS] and [SEP] tokens. [SEP] is simply EOS, or </S> token.
            caption_tokens.insert(0, self._vocabulary.cls_token)
            caption_tokens.append(self._vocabulary.sep_token)

            # Trim captions up to maximum length.
            caption_tokens = caption_tokens[: self._max_caption_length]

            # Pad the sequence of tokens up to maximum length.
            # This makes the default ``collate_fn`` of dataloader work.
            caption_tokens.extend(
                [self._vocabulary.pad_token]
                * (self._max_caption_length - len(caption_tokens))
            )
            # Convert (string) tokens to (integer) token indices.
            token_indices = [
                self._vocabulary.get_token_index(t) for t in caption_tokens
            ]

            # We refer "token indices" as "tokens" after this point (in the
            # model and such), because how could they be string tensors?!
            datapoint["caption_tokens"] = token_indices

            yield datapoint
