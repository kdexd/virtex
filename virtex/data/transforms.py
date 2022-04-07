import albumentations as alb
import cv2


class HorizontalFlip(alb.BasicTransform):
    r"""
    Flip the image horizontally randomly (equally likely) and replace the
    word "left" with "right" in the caption.

    .. note::

        This transform can also work on images only (without the captions).
        Its behavior will be same as albumentations
        :class:`~albumentations.augmentations.transforms.HorizontalFlip`.

    Examples:
        >>> flip = HorizontalFlip(p=0.5)
        >>> out1 = flip(image=image, caption=caption)  # keys: {"image", "caption"}
        >>> # Also works with images (without caption).
        >>> out2 = flip(image=image)  # keys: {"image"}

    """

    @property
    def targets(self):
        return {"image": self.apply, "caption": self.apply_to_caption}

    def apply(self, img, **params):
        return cv2.flip(img, 1)

    def apply_to_caption(self, caption, **params):
        caption = (
            caption.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
        return caption


class RandomResizedSquareCrop(alb.RandomResizedCrop):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.RandomResizedCrop`
    which assumes a square crop (width = height). Everything else is same.

    Args:
        size: Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


class CenterSquareCrop(alb.CenterCrop):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.CenterCrop`
    which assumes a square crop (width = height). Everything else is same.

    Args:
        size: Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


class SquareResize(alb.Resize):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.Resize` which
    assumes a square resize (width = height). Everything else is same.

    Args:
        size: Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


# =============================================================================
#   SOME COMMON CONSTANTS AND IMAGE TRANSFORMS:
#   These serve as references here, and are used as default params in many
#   dataset class constructors.
# -----------------------------------------------------------------------------

IMAGENET_COLOR_MEAN = (0.485, 0.456, 0.406)
r"""ImageNet color normalization mean in RGB format (values in 0-1)."""

IMAGENET_COLOR_STD = (0.229, 0.224, 0.225)
r"""ImageNet color normalization std in RGB format (values in 0-1)."""

DEFAULT_IMAGE_TRANSFORM = alb.Compose(
    [
        alb.SmallestMaxSize(256, p=1.0),
        CenterSquareCrop(224, p=1.0),
        alb.Normalize(mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0),
    ]
)
r"""Default transform without any data augmentation (during pretraining)."""
# =============================================================================
