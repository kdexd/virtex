r"""
This module contains a bunch of dict-like data structures used by datasets in
:mod:`virtex.data.datasets` for returning instances and batches of training
(or inference) data.

These classes are thin wrappers over native python dicts for two main purposes:

1. Better readability, type-hint annotations and stronger static type checking.
2. Each of these classes implement ``__slots__``: a trick to significantly
   reduce memory overhead in creating hundreds of new dict objects (every
   iteration) if the name of dict keys remain fixed (and unchanged).

"""
import copy
from typing import Iterable, List, Optional, Union

import torch


class Instance(dict):
    r"""
    Base class for representing a single instance: a dict of key value pairs.
    Keys are assumed to be ``str``, and values are assumed to be :class:`torch.Tensor`.

    This class can be instantiated with ``**kwargs`` just like ``dict()``.
    """

    def to(self, *args, **kwargs) -> "Instance":
        r"""
        Defines the logic to move the whole instance across different
        :class:`torch.dtype`s and :class:`torch.device`s. Default implementation
        shifts all tensor-like objects to device.

        .. note::

            This method is used internally by `NVIDIA Apex <https://github.com/nvidia/apex>`_
            for casting the instance to FP16 (half) precision -- this method
            casts floats to half; while keeping integers, booleans and other
            data types unchanged.
        """
        new_instance = self.clone()
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        # Casting to non-float dtype is not allowed. Common cast dtype is
        # `torch.half`, which would be done internally by NVIDIA Apex for mixed
        # precision training.
        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError(
                    f"Can cast {self.__class__.__name__} to a floating point "
                    f"dtype, but got desired dtype={dtype}"
                )
            else:
                # Cast all members which are of floating point dtype.
                for key in new_instance.keys():
                    if new_instance[key].dtype.is_floating_point:
                        new_instance[key] = new_instance[key].to(dtype)

        # Transfer all tensors to a specific device.
        if device is not None:
            for key in new_instance.keys():
                new_instance[key] = new_instance[key].to(device)

        return new_instance

    def pin_memory(self):
        r"""Pin GPU memory; only used internally by PyTorch dataloaders."""
        for key in self.keys():
            self[key].pin_memory()

    def clone(self) -> "Instance":
        return copy.deepcopy(self)


class Batch(dict):
    r"""
    Base class for representing a single batch. It is created with a list of
    instances; and used by ``collate_fn`` of :mod:`virtex.data.datasets`. This
    is exactly same as :class:`~virtex.data.structures.Instance` in terms of
    behavior and functionality.
    """

    def to(self, *args, **kwargs) -> "Batch":
        new_batch = self.clone()
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError(
                    f"Can cast {self.__class__.__name__} to a floating point "
                    f"dtype, but got desired dtype={dtype}"
                )
            else:
                for key in new_batch.keys():
                    if new_batch[key].dtype.is_floating_point:
                        new_batch[key] = new_batch[key].to(dtype)

        if device is not None:
            for key in new_batch.keys():
                new_batch[key] = new_batch[key].to(device)
        return new_batch

    def pin_memory(self):
        for key in self.keys():
            self[key].pin_memory()

    def clone(self) -> "Batch":
        return copy.deepcopy(self)


class ImageCaptionInstance(Instance):
    r"""
    An instance representing an image-caption pair. It contains caption tokens
    in both, forward and backward direction.

    Member names: ``{"image_id", "image", "caption_tokens", "noitpac_tokens",
    "caption_lengths"}``

    Parameters
    ----------
    image_id: int
        A unique integer ID for current image (or instance). This is commonly
        the COCO image ID.
    image: Iterable[float]
        Image tensor (or numpy array) in CHW format.
    caption_tokens: List[int]
        Tokenized caption sequences.
    """

    __slots__ = [
        "image_id",
        "image",
        "caption_tokens",
        "noitpac_tokens",
        "caption_lengths",
    ]

    def __init__(
        self, image_id: int, image: Iterable[float], caption_tokens: List[int]
    ):
        super().__init__(
            image_id=torch.tensor(image_id, dtype=torch.long),
            image=torch.tensor(image, dtype=torch.float),
            caption_tokens=torch.tensor(caption_tokens, dtype=torch.long),
            noitpac_tokens=torch.tensor(caption_tokens, dtype=torch.long).flip(0),
            caption_lengths=torch.tensor(len(caption_tokens), dtype=torch.long),
        )


class ImageCaptionBatch(Batch):
    r"""
    Batch of :class:`~virtex.data.structures.ImageCaptionInstance`. Contains
    same keys as instances.

    Parameters
    ----------
    instances: List[ImageCaptionInstance]
        List of :class:`~virtex.data.structures.ImageCaptionInstance` to be
        collated into a batch.
    padding_value: int, optional (default = 0)
        Padding value to fill while batching captions of different lengths.
    """

    __slots__ = [
        "image_id",
        "image",
        "caption_tokens",
        "noitpac_tokens",
        "caption_lengths",
    ]

    def __init__(
        self, instances: List[ImageCaptionInstance], padding_value: int = 0
    ):

        # Stack `image_id` and `image` from instances to create batch at dim 0.
        image_id = torch.stack([ins["image_id"] for ins in instances], dim=0)
        image = torch.stack([ins["image"] for ins in instances], dim=0)

        if "caption_tokens" in instances[0]:
            # Find maximum caption length in this batch.
            max_caption_length = max([ins["caption_lengths"] for ins in instances])

            # Pad `caption_tokens` and `masked_labels` up to this length.
            caption_tokens = torch.nn.utils.rnn.pad_sequence(
                [ins["caption_tokens"] for ins in instances],
                batch_first=True,
                padding_value=padding_value,
            )
            noitpac_tokens = torch.nn.utils.rnn.pad_sequence(
                [ins["noitpac_tokens"] for ins in instances],
                batch_first=True,
                padding_value=padding_value,
            )
            caption_lengths = torch.stack(
                [ins["caption_lengths"] for ins in instances]
            )

            super().__init__(
                image_id=image_id,
                image=image,
                caption_tokens=caption_tokens,
                noitpac_tokens=noitpac_tokens,
                caption_lengths=caption_lengths,
            )
        else:
            super().__init__(image_id=image_id, image=image)


class MaskedLmInstance(Instance):

    __slots__ = [
        "image_id",
        "image",
        "caption_tokens",
        "caption_lengths",
        "masked_labels",
    ]

    def __init__(
        self,
        image_id: int,
        image: Iterable[float],
        caption_tokens: List[int],
        masked_labels: List[int],
    ):
        super().__init__(
            image_id=torch.tensor(image_id, dtype=torch.long),
            image=torch.tensor(image, dtype=torch.float),
            caption_tokens=torch.tensor(caption_tokens, dtype=torch.long),
            caption_lengths=torch.tensor(len(caption_tokens), dtype=torch.long),
            masked_labels=torch.tensor(masked_labels, dtype=torch.long),
        )


class MaskedLmBatch(Batch):

    __slots__ = [
        "image_id",
        "image",
        "caption_tokens",
        "caption_lengths",
        "masked_labels",
    ]

    def __init__(self, instances: List[MaskedLmInstance], padding_value: int = 0):

        # Stack `image_id` and `image` from instances to create batch at dim 0.
        image_id = torch.stack([ins["image_id"] for ins in instances], dim=0)
        image = torch.stack([ins["image"] for ins in instances], dim=0)

        # Find maximum caption length in this batch.
        max_caption_length = max([ins["caption_lengths"] for ins in instances])

        # Pad `caption_tokens` and `masked_labels` up to this length.
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [ins["caption_tokens"] for ins in instances],
            batch_first=True,
            padding_value=padding_value,
        )
        masked_labels = torch.nn.utils.rnn.pad_sequence(
            [ins["masked_labels"] for ins in instances],
            batch_first=True,
            padding_value=padding_value,
        )
        caption_lengths = torch.stack([ins["caption_lengths"] for ins in instances])

        super().__init__(
            image_id=image_id,
            image=image,
            caption_tokens=caption_tokens,
            caption_lengths=caption_lengths,
            masked_labels=masked_labels,
        )


class LinearClassificationInstance(Instance):
    r"""
    An instance representing an image-label pair.

    Member names: ``{"image_id", "label"}``

    Parameters
    ----------
    image: Iterable[float]
        Image tensor (or numpy array) in CHW format.
    label: Union[int, torch.Tensor]
        An integer (or one-hot vector) label corresponding to the image.
    """

    __slots__ = ["image", "label"]

    def __init__(self, image: Iterable[float], label: Union[int, torch.Tensor]):
        super().__init__(
            image=torch.tensor(image, dtype=torch.float),
            label=torch.tensor(label, dtype=torch.long),
        )


class LinearClassificationBatch(Batch):
    r"""
    Batch of :class:`~virtex.data.structures.LinearClassificationInstance`.
    Contains same keys as instances.

    Parameters
    ----------
    instances: List[ImageCaptionInstance]
        List of :class:`~virtex.data.structures.LinearClassificationInstance`
        to be collated into a batch.
    """

    __slots__ = ["image", "label"]

    def __init__(self, instances: List[LinearClassificationInstance]):

        # Stack `image` and `label` from instances to create batch at dim 0.
        image = torch.stack([ins["image"] for ins in instances], dim=0)
        label = torch.stack([ins["label"] for ins in instances], dim=0)
        super().__init__(image=image, label=label)
