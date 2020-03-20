from typing import List
import unicodedata

import albumentations as alb
import tokenizers as tkz
import numpy as np


# ImageNet color normalization mean and std in RGB format (values in 0-1).
IMAGENET_COLOR_MEAN = (0.485, 0.456, 0.406)
IMAGENET_COLOR_STD = (0.229, 0.224, 0.225)


class CaptionOnlyTransform(alb.BasicTransform):
    r"""
    Transforms in :mod:`albumentations` are mainly suited for images (and bbox,
    keypoints etc.) We extend :class:`albumentations.core.transforms_interface.BasicTransform`
    to transform captions. Captions may be ``str``, or tokens (``List[int]``)
    as per implementation of :meth:`apply_to_caption`. These transforms will
    have consistent API as other transforms from albumentations.
    """

    @property
    def targets(self):
        return {"caption": self.apply_to_caption}

    def apply_to_caption(self, caption, **params):
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        # Super class adds "width" and "height" but we don't have image here.
        return params


class ImageCaptionTransform(alb.BasicTransform):
    r"""
    Similar to :class:`CaptionOnlyTranform`, this extends super class to work
    on ``(image, caption)`` pair together.
    """

    @property
    def targets(self):
        return {"image": self.apply, "caption": self.apply_to_caption}

    def apply_to_caption(self):
        raise NotImplementedError


class NormalizeCaption(CaptionOnlyTransform):
    r"""
    Perform common normalization with caption -- lowercase, trim leading and
    trailing whitespaces, NFKD normalization and strip accents.

    Examples
    --------
    >>> normalize = NormalizeCaption(always_apply=True)
    >>> out = normalize(caption="Some caption input here.")  # keys: {"caption"}
    """

    def __init__(self):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)

    def apply_to_caption(self, caption, **params):
        caption = caption.lower()
        caption = unicodedata.normalize("NFKD", caption)
        caption = "".join(
            [chr for chr in caption if not unicodedata.combining(chr)]
        )
        return caption


class TokenizeCaption(CaptionOnlyTransform):
    r"""Tokenize a caption (``str``) to list of tokens (``int``)."""

    def __init__(self, tokenizer: tkz.implementations.BaseTokenizer):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)
        self._tokenizer = tokenizer

    def apply_to_caption(self, caption, **params):
        token_indices: List[int] = self._tokenizer.encode(caption).ids

        # Add boundary tokens.
        token_indices.insert(0, self._tokenizer.token_to_id("[SOS]"))
        token_indices.append(self._tokenizer.token_to_id("[EOS]"))
        return token_indices

    def get_transform_init_args_names(self):
        return ("tokenizer",)


class TruncateCaptionTokens(CaptionOnlyTransform):
    r"""Truncate a list of caption tokens (``int``) to maximum length."""

    def __init__(self, max_caption_length: int = 30):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)
        self.max_caption_length = max_caption_length

    def apply_to_caption(self, caption, **params):
        return caption[: self.max_caption_length]

    def get_transform_init_args_names(self):
        return ("max_caption_length",)


class RandomHorizontalFlip(ImageCaptionTransform):
    r"""
    Flip the image horizontally randomly (equally likely) and replace the
    word "left" with "right" in the caption.

    Examples
    --------
    >>> flip = RandomHorizontalFlip(p=0.5)
    >>> out = flip(image=image, caption=caption)  # keys: {"image", "caption"}
    """

    def apply(self, img, **params):
        image = np.ascontiguousarray(img[:, ::-1, ...])
        return image

    def apply_to_caption(self, caption, **params):
        caption = (
            caption.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
        return caption


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
        return ("alpha",)
