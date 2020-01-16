import math
import random
from typing import List, Union

import dataflow as df
from dataflow import imgaug as aug
import numpy as np

from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary


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
        image_resize_size: int = 256,
        image_crop_size: int = 224,
        index_or_key: Union[int, str] = "image",
    ):
        self.ds = ds
        self._normalize = normalize
        self._x = index_or_key

        self._augmentor = df.imgaug.AugmentorList(
            [
                aug.ResizeShortestEdge(image_resize_size),
                aug.RandomCrop(image_crop_size),
                aug.ToFloat32(),
                aug.MapImage(self._transform),
            ]
        )

    def _transform(self, image: np.ndarray) -> np.ndarray:
        image = image / 255.0
        if self._normalize:
            image -= np.array([0.485, 0.456, 0.406])
            image /= np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))
        return image

    def __iter__(self):
        for datapoint in self.ds:
            image = self._augmentor.augment(datapoint[self._x])
            datapoint[self._x] = image
            yield datapoint


class RandomHorizontalFlip(df.ProxyDataFlow):
    r"""
    Flip the image horizontally randomly (equally likely) and replace the
    word "left" with "right" in the caption. In the data-loading pipeline, put
    this after :class:`TransformImageForResnetLikeModels`.
    """

    def __init__(
        self, ds: df.DataFlow, image_key: str = "image", caption_key: str = "caption"
    ):
        self.ds = ds
        self._i = image_key
        self._c = caption_key

    def __iter__(self):
        for datapoint in self.ds:
            flag = random.random()
            if flag < 0.5:
                datapoint[self._i] = np.ascontiguousarray(
                    datapoint[self._i][:, ::-1, ...]
                )
                caption = datapoint[self._c]

                # Interchange words "left" and "right" if flipped.
                caption = (
                    caption.replace("left", "__TEMP__")
                    .replace("right", "left")
                    .replace("__TEMP__", "right")
                )
                datapoint[self._c] = caption
            yield datapoint


class TokenizeCaption(df.ProxyDataFlow):
    def __init__(
        self,
        ds: df.DataFlow,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        max_caption_length: int = 25,
        input_key: str = "caption",
        output_key: str = "caption_tokens",
    ):
        self.ds = ds
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._ik = input_key
        self._ok = output_key
        self._max_caption_length = max_caption_length

    def __iter__(self):

        for datapoint in self.ds:
            caption = datapoint[self._ik]

            # Tokenize caption (these are still strings).
            caption = datapoint[self._ik]
            caption_tokens = self._tokenizer.tokenize(caption)

            # Add [CLS] and [SEP] tokens. [SEP] is simply EOS, or </S> token.
            caption_tokens.insert(0, self._vocabulary.cls_token)
            caption_tokens.append(self._vocabulary.sep_token)

            # Trim captions up to maximum length.
            caption_tokens = caption_tokens[: self._max_caption_length]

            # Convert (string) tokens to (integer) token indices.
            token_indices = [
                self._vocabulary.get_token_index(t) for t in caption_tokens
            ]
            datapoint[self._ok] = token_indices

            yield datapoint


class MaskSomeTokensRandomly(df.ProxyDataFlow):
    r"""
    Mask some tokens randomly tokens from an input caption. Do this before
    padding the caption to maximum length to avoid instance with no padded tokens.
    """

    # Make sure to change here if changed in SentencePieceTokenizer.
    SP_SPACE = u"▁"

    def __init__(
        self,
        ds: df.DataFlow,
        vocabulary: SentencePieceVocabulary,
        mask_proportion: float = 0.15,
        mask_probability: float = 0.80,
        replace_probability: float = 0.10,
        input_key: str = "masked_tokens",
        output_key: str = "masked_labels",
    ):
        self.ds = ds
        self._vocabulary = vocabulary

        self._mask_index = vocabulary.mask_index
        self._pad_index = vocabulary.pad_index
        self._mask_proportion = mask_proportion
        self._mask_prob = mask_probability
        self._repl_prob = replace_probability
        self._ik = input_key
        self._ok = output_key

    def __iter__(self):

        for datapoint in self.ds:
            masked_tokens = datapoint[self._ik]
            masked_labels = [self._pad_index] * len(masked_tokens)

            # Indices in `masked_tokens` list to mask (minimum 1 index).
            # Leave out first and last indices (boundary tokens).
            tokens_to_mask: List[int] = random.sample(
                list(range(1, len(masked_tokens) - 1)),
                math.ceil((len(masked_tokens) - 2) * self._mask_proportion),
            )
            for i in tokens_to_mask:
                # Whether to replace with [MASK] or random word.
                # If only one token, always [MASK].
                if len(tokens_to_mask) == 1:
                    masked_labels[i] = masked_tokens[i]
                    masked_tokens[i] = self._mask_index
                else:
                    _flag: float = random.random()
                    if _flag <= self._mask_prob + self._repl_prob:
                        if _flag <= self._mask_prob:
                            masked_labels[i] = masked_tokens[i]
                            masked_tokens[i] = self._mask_index
                        else:
                            masked_tokens[i] = self._random_token_index()

            # At this point, caption tokens and masked labels are lists of
            # same length. Do whole word masking now.
            for i in range(len(masked_tokens)):
                if masked_tokens[i] == self._mask_index:
                    # Mask all following tokens until getting one which starts
                    # with a space.
                    for j in range(i + 1, len(masked_tokens)):
                        tt = self._vocabulary.get_token_from_index(masked_tokens[j])
                        if (
                            tt.startswith(self.SP_SPACE)
                            or tt in self._vocabulary.special_tokens
                        ):
                            break
                        masked_labels[j] = masked_tokens[j]
                        masked_tokens[j] = self._mask_index

                    # Mask tokens before this one, if this one doesn't start
                    # with a space.
                    t = self._vocabulary.get_token_from_index(masked_labels[i])
                    if (
                        not t.startswith(self.SP_SPACE)
                        and t not in self._vocabulary.special_tokens
                    ):
                        for j in range(i - 1, -1, -1):
                            tt = self._vocabulary.get_token_from_index(
                                masked_tokens[j]
                            )
                            if tt in self._vocabulary.special_tokens:
                                break
                            if tt.startswith(self.SP_SPACE):
                                masked_labels[j] = masked_tokens[j]
                                masked_tokens[j] = self._mask_index
                                break
                            masked_labels[j] = masked_tokens[j]
                            masked_tokens[j] = self._mask_index

            datapoint[self._ik] = masked_tokens
            datapoint[self._ok] = masked_labels
            yield datapoint

    def _random_token_index(self) -> int:
        while True:
            token_index = random.randint(0, len(self._vocabulary) - 1)
            if token_index not in self._vocabulary.special_indices:
                return token_index
