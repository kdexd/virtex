import random

import albumentations as alb
import dataflow as df
import numpy as np

from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary


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
                # Flip the "width" axis for image in HWC format.
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
        input_key: str = "caption",
        output_key: str = "caption_tokens",
    ):
        self.ds = ds
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._ik = input_key
        self._ok = output_key

    def __iter__(self):

        for datapoint in self.ds:
            caption = datapoint[self._ik]

            # Tokenize caption (these are still strings).
            caption = datapoint[self._ik]
            caption_tokens = self._tokenizer.tokenize(caption)

            # Add [CLS] and [SEP] tokens. [SEP] is simply EOS, or </S> token.
            caption_tokens.insert(0, self._vocabulary.cls_token)
            caption_tokens.append(self._vocabulary.sep_token)

            # Convert (string) tokens to (integer) token indices.
            token_indices = [
                self._vocabulary.get_token_index(t) for t in caption_tokens
            ]
            datapoint[self._ok] = token_indices

            yield datapoint


class AlexNetPCA(alb.ImageOnlyTransform):
    r"""
    Lighting noise(AlexNet - style PCA - based noise). This trick was
    originally used in `AlexNet paper <https://papers.nips.cc/paper/4824-imagenet-classification
    -with-deep-convolutional-neural-networks.pdf>`_

    The eigen values and eigen vectors, are taken from caffe2 `ImageInputOp.h
    <https://github.com/pytorch/pytorch/blob/master/caffe2/image/image_input_op.h#L265>`_.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.eigval = np.array([[0.2175], [0.0188], [0.0045]])
        self.eigvec = np.array(
            [
                [-144.7125, 183.396, 102.2295],
                [-148.104, -1.1475, -207.57],
                [-148.818, -177.174, 107.1765],
            ]
        )

    def apply(self, img, **params):
        alpha = np.random.normal(0.0, self.alpha, size=(3, 1)) * self.eigval
        add_vector = np.matrix(self.eigvec) * np.matrix(alpha)

        img = img + add_vector[np.newaxis, np.newaxis, ...]
        img = np.clip(img, 0.0, 255.0)
        return img

    def get_transform_init_args_names(self):
        return "alpha",
