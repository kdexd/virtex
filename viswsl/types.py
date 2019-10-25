from typing import Sequence

from mypy_extensions import TypedDict
import numpy as np


class LmdbDatapoint(TypedDict, total=False):
    image_id: int
    image: np.ndarray
    captions: Sequence[str]
    caption_tokens: Sequence[int]


class MaskedLanguageModelingInstance(TypedDict, total=True):
    image_id: int
    image: np.ndarray
    captions: Sequence[int]
    caption_tokens: Sequence[int]
    masked_labels: Sequence[int]
