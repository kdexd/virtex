import random
from typing import List
import unicodedata

import albumentations as alb
import cv2
import numpy as np

from viswsl.data.tokenizer import SentencePieceBPETokenizer


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
        caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])
        return caption


class TokenizeCaption(CaptionOnlyTransform):
    r"""Tokenize a caption (``str``) to list of tokens (``int``)."""

    def __init__(self, tokenizer: SentencePieceBPETokenizer):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)
        self._tokenizer = tokenizer

    def apply_to_caption(self, caption, **params):
        token_indices: List[int] = self._tokenizer.encode(caption)

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


class HorizontalFlip(ImageCaptionTransform):
    r"""
    Flip the image horizontally randomly (equally likely) and replace the
    word "left" with "right" in the caption. This transform can also work on
    images only (without the captions).

    Examples
    --------
    >>> flip = ImageCaptionHorizontalFlip(p=0.5)
    >>> out = flip(image=image, caption=caption)  # keys: {"image", "caption"}
    """

    def apply(self, img, **params):
        return cv2.flip(img, 1)

    def apply_to_caption(self, caption, **params):
        caption = (
            caption.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
        return caption


class ColorJitter(alb.ImageOnlyTransform):
    r"""
    Randomly change brightness, contrast, hue and saturation of the image. This
    class behaves exactly like :class:`torchvision.transforms.ColorJitter` but
    is slightly faster (uses OpenCV) and compatible with rest of the transforms
    used here (albumentations-style). This class works only on ``uint8`` images.

    .. note::

        Unlike torchvision variant, this class follows "garbage-in, garbage-out"
        policy and does not check limits for jitter factors. User must ensure
        that ``brightness``, ``contrast``, ``saturation`` should be ``float``s
        ``[0, 1]`` and ``hue`` should be a ``float`` in ``[0, 0.5]``.

    Parameters
    ----------
    brightness: float, optional (default = 0)
        How much to jitter brightness. ``brightness_factor`` is chosen
        uniformly from ``[1 - brightness, 1 + brightness]``.
    contrast: float, optional (default = 0)
        How much to jitter contrast. ``contrast_factor`` is chosen uniformly
        from ``[1 - contrast, 1 + contrast]``
    saturation: float, optional (default = 0)
        How much to jitter saturation. ``saturation_factor`` is chosen
        uniformly from ``[1 - saturation, 1 + saturation]``.
    hue: float, optional (default = 0)
        How much to jitter hue. ``hue_factor`` is chosen uniformly from
        ``[-hue, hue]``.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, img, **params):
        original_dtype = img.dtype

        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        # Convert arguments as required by albumentations functional interface.
        # "gain" = contrast and "bias" = (brightness_factor - 1)
        img = alb.augmentations.functional.brightness_contrast_adjust(
            img, alpha=contrast_factor, beta=brightness_factor - 1
        )
        # Hue and saturation limits are required to be integers.
        img = alb.augmentations.functional.shift_hsv(
            img,
            hue_shift=int(hue_factor * 255),
            sat_shift=int(saturation_factor * 255),
        )
        img = img.astype(original_dtype)
        return img

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "saturation", "hue")


class LightingNoise(alb.ImageOnlyTransform):
    r"""
    Lighting noise (AlexNet - style PCA - based noise). This trick was
    originally used in `AlexNet paper <https://papers.nips.cc/paper/4824-imagenet-classification
    -with-deep-convolutional-neural-networks.pdf>`_
    The eigen values and eigen vectors, are taken from caffe2 `ImageInputOp.h
    <https://github.com/pytorch/pytorch/blob/master/caffe2/image/image_input_op.h#L265>`_.

    This class only works for ``uint8`` images.
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
        original_dtype = img.dtype

        alpha = np.random.normal(0.0, self.alpha, size=(3, 1)) * self.eigval
        add_vector = np.matrix(self.eigvec) * np.matrix(alpha)

        img = img + np.asarray(add_vector).squeeze()[np.newaxis, np.newaxis, ...]
        img = np.clip(img, 0.0, 255.0)

        img = img.astype(original_dtype)
        return img

    def get_transform_init_args_names(self):
        return ("alpha",)


class RandomResizedSquareCrop(alb.RandomResizedCrop):
    r"""
    A variant of :class:`albumentations.transforms.RandomResizedCrop` which
    assumes a square crop (width = height). Everything else is same.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


class CenterSquareCrop(alb.CenterCrop):
    r"""
    A variant of :class:`albumentations.transforms.CenterCrop` which
    assumes a square crop (width = height). Everything else is same.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


# =============================================================================
#   SOME COMMON CONSTANTS AND IMAGE TRANSFORMS:
#   These serve as references here, and are used as default params in many
#   dataset class constructors.
# -----------------------------------------------------------------------------

r"""ImageNet color normalization mean in RGB format (values in 0-1)."""
IMAGENET_COLOR_MEAN = (0.485, 0.456, 0.406)

r"""ImageNet color normalization std in RGB format (values in 0-1)."""
IMAGENET_COLOR_STD = (0.229, 0.224, 0.225)

r"""Default transform without any data augmentation (during pretraining)."""
DEFAULT_IMAGE_TRANSFORM = alb.Compose(
    [
        alb.SmallestMaxSize(256, p=1.0),
        CenterSquareCrop(224, p=1.0),
        alb.Normalize(mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0),
    ]
)

r"""
Desired image transform for ImageNet and Places205 linear classification
protocol. This follows evaluation protocol similar to several prior works::

    1. `(Misra et al, 2019) "Self-Supervised Learning of Pretext-Invariant
       Representations" <https://arxiv.org/abs/1912.01991>`_.

    2. `(Goyal et al, 2019) "Scaling and Benchmarking Self-Supervised Visual
       Representation Learning" <https://arxiv.org/abs/1905.01235>`_.

    3. `(Zhang et al, 2016a) "Colorful Image Colorization" 
       <https://arxiv.org/abs/1603.08511>`_.

    4. `(Zhang et al, 2016b) "Split-Brain Autoencoders: Unsupervised Learning
       by Cross-Channel Prediction" <https://arxiv.org/abs/1611.09842>`_.

This should ideally not be changed for apples-to-apples comparison.
"""
LINEAR_CLF_IMAGE_TRANSFORM_TRAIN = alb.Compose(
    [
        RandomResizedSquareCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33), p=1.0),
        HorizontalFlip(p=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5),
        alb.Normalize(mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0),
    ]
)

r"""
Desired image transform for ImageNet and Places205 linear classification
protocol during validation phase. Consistent with prior works listed above.
"""
LINEAR_CLF_IMAGE_TRANSFORM_VAL = alb.Compose(
    [
        alb.SmallestMaxSize(256, p=1.0),
        CenterSquareCrop(224, p=1.0),
        alb.Normalize(mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0),
    ]
)

r"""
Desired image transform for VOC2007 linear classification protocol. This
follows evaluation protocol as introduced in `(Goyal et al, 2019) "Scaling and
Benchmarking Self-Supervised Visual Representation Learning" <https://arxiv.org/abs/1905.01235>`_.
This should ideally not be changed for apples-to-apples comparison.
"""
VOC07_CLF_IMAGE_TRANSFORM = alb.Compose(
    [
        alb.Resize(224, 224, p=1.0),
        alb.Normalize(mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0),
    ]
)
# =============================================================================
