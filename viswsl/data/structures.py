import copy
from typing import Iterable, List

import torch


class Instance(dict):
    def to(self, *args, **kwargs) -> "Instance":
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

    def clone(self) -> "Instance":
        return copy.deepcopy(self)


class Batch(dict):
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

    def clone(self) -> "Batch":
        return copy.deepcopy(self)


class WordMaskingInstance(Instance):

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


class WordMaskingBatch(Batch):

    __slots__ = [
        "image_id",
        "image",
        "caption_tokens",
        "caption_lengths",
        "masked_labels",
    ]

    def __init__(self, instances: List[WordMaskingInstance], padding_value: int = 0):

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


class CaptioningInstance(Instance):

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


class CaptioningBatch(Batch):

    __slots__ = [
        "image_id",
        "image",
        "caption_tokens",
        "noitpac_tokens",
        "caption_lengths",
    ]

    def __init__(self, instances: List[CaptioningInstance], padding_value: int = 0):

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
        noitpac_tokens = torch.nn.utils.rnn.pad_sequence(
            [ins["noitpac_tokens"] for ins in instances],
            batch_first=True,
            padding_value=padding_value,
        )
        caption_lengths = torch.stack([ins["caption_lengths"] for ins in instances])

        super().__init__(
            image_id=image_id,
            image=image,
            caption_tokens=caption_tokens,
            noitpac_tokens=noitpac_tokens,
            caption_lengths=caption_lengths,
        )
