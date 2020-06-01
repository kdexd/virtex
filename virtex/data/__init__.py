from .datasets.captioning import CaptioningDataset
from .datasets.multilabel import MultiLabelClassificationDataset
from .datasets.downstream import (
    ImageNetDataset,
    INaturalist2018Dataset,
    VOC07ClassificationDataset,
    CocoCaptionsEvalDataset,
)

__all__ = [
    "CaptioningDataset",
    "MultiLabelClassificationDataset",
    "CocoCaptionsEvalDataset",
    "ImageNetDataset",
    "INaturalist2018Dataset",
    "VOC07ClassificationDataset",
]
