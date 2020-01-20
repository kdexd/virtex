import random
from typing import List
import unicodedata

import albumentations as alb
import dataflow as df
import tokenizers as tkz
import numpy as np


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
        tokenizer: tkz.implementations.BaseTokenizer,
        input_key: str = "caption",
        output_key: str = "caption_tokens",
    ):
        self.ds = ds
        self._tokenizer = tokenizer

        # Short handle for convenience
        self._sos_index = self._tokenizer.token_to_id("[SOS]")
        self._eos_index = self._tokenizer.token_to_id("[EOS]")

        self._ik = input_key
        self._ok = output_key

    def __iter__(self):

        for datapoint in self.ds:
            caption = datapoint[self._ik]

            # Lowercase caption and strip accents from characters.
            caption = caption.lower()
            caption = unicodedata.normalize("NFKD", caption)
            caption = "".join(
                [chr for chr in caption if not unicodedata.combining(chr)]
            )
            token_indices: List[int] = self._tokenizer.encode(caption).ids

            # Add boundary tokens, we use same token for start and end.
            token_indices.insert(0, self._sos_index)
            token_indices.append(self._eos_index)
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

    def __init__(self, alpha: float = 0.1, p: float = 0.5):
        super().__init__(p=p)
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

        img = img + np.asarray(add_vector).squeeze()[np.newaxis, np.newaxis, ...]
        img = np.clip(img, 0.0, 255.0)
        return img

    def get_transform_init_args_names(self):
        return "alpha",
