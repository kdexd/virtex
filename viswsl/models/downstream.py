from typing import Any, Dict

import torch
from torch import nn

from viswsl.utils.metrics import TopkAccuracy


class FeatureExtractor(nn.Module):
    r"""
    Extract features from intermediate ResNet layers (stages) such that their
    feature dimension is approximately 9000. These features will be used to
    train SVMs / linear layers to evaluate quality of learned representations.

    This evaluation protocol is consistent with FAIR Self Supervision Benchmark
    `(Goyal et al, 2019) <https://arxiv.org/abs/1905.01235>`_, Colorful Image
    Colorization `(Zhang et al, 2016a) <https://arxiv.org/abs/1603.08511>`_ and
    Split-Brain Autoencoder `(Zhang et al, 2016b) <https://arxiv.org/abs/1611.09842>`_.

    Parameters
    ----------
    trained_model: nn.Module
        Trained model (either imagenet or one of our pretext tasks). We would
        only use the visual stream.
    layer_name: str, optional (default = "layer4")
        Which layer of ResNet to extract features from. One of ``{"layer1",
        "layer2", "layer3", "layer4", "avgpool"}``.
    flatten_and_normalize: bool, optional (default = False)
        Whether to flatten the features and perform L2 normalization. This flag
        is ``True`` for VOC07 linear SVM classification, ``False`` otherwise.
    """

    def __init__(
        self,
        trained_model: nn.Module,
        layer_name: str = "layer4",
        flatten_and_normalize: bool = False,
    ):
        super().__init__()
        self.visual: nn.Module = trained_model.visual.eval()  # type: ignore

        # Check if layer name is valid.
        if layer_name not in {"layer1", "layer2", "layer3", "layer4", "avgpool"}:
            raise ValueError(f"Invalid layer name: {layer_name}")

        pool_and_feature_sizes = {
            "layer1": (6, 256 * 6 * 6),
            "layer2": (4, 512 * 4 * 4),
            "layer3": (3, 1024 * 3 * 3),
            "layer4": (2, 2048 * 2 * 2),
            "avgpool": (1, 2048 * 1 * 1),
        }
        # This pool layer will downsample features from ResNet-like models
        # so their size is ~9000 when flattened.
        if "layer" in layer_name:
            self.pool = nn.AdaptiveAvgPool2d(pool_and_feature_sizes[layer_name][0])
        else:
            self.pool = nn.Identity()

        self.layer_name = layer_name
        self.feature_size = pool_and_feature_sizes[layer_name][1]
        self.flatten_and_normalize = flatten_and_normalize

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        features = self.visual(images, return_intermediate_outputs=True)
        pooled = self.pool(features[self.layer_name])

        # Perform normalization and flattening of features.
        if self.flatten_and_normalize:
            pooled = pooled.view(pooled.size(0), -1)
            pooled = pooled / torch.norm(pooled, dim=-1).unsqueeze(-1)

        return pooled


class LinearClassificationProtocolModel(nn.Module):
    r"""
    A combination of (BN + linear) layer for classification of features
    extracted from intermediate ResNet layers. This is used for ImageNet and
    Places205 datasets. This module computes cross entropy loss during forward
    pass, and also accumulates Top-1 accuracy during validation.

    Parameters
    ----------
    feature_extractor: FeatureExtractor
        Feature extractor to train linear classifier on.
    num_classes: int, optional (default = 1000)
        Number of output classes (for softmax). Set to 1000 for ImageNet and
        205 for Places205.
    norm_layer: str, optional (default = "FrozenBN")
        Which norm layer to use.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        num_classes: int = 1000,
        norm_layer: str = "FrozenBN",
    ):
        super().__init__()
        self.layer_name = feature_extractor.layer_name

        # BN + linear layer for extracted features.
        self.bn = (
            nn.BatchNorm2d(2048, affine=True)
            if norm_layer == "BN"
            else nn.BatchNorm2d(2048, affine=False)
            if norm_layer == "FrozenBN"
            else nn.Identity()
        )
        self.fc = nn.Linear(feature_extractor.feature_size, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.top1 = TopkAccuracy(top_k=1)

        torch.nn.init.normal_(self.fc.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:

        # shape: (batch_size, num_classes)
        batch_size = features.size(0)
        features = self.bn(features)

        # Flatten the normlized features before passing to fc layer.
        features = features.view(batch_size, -1)
        logits = self.fc(features)

        loss = self.loss(logits, labels)

        # Calculate cross entropy loss using `labels`. Keep dict structure like
        # this (despite one key) for convenient Tensorboard logging.
        output_dict: Dict[str, Any] = {"loss": {self.layer_name: loss}}

        if not self.training:
            # shape: (batch_size, )
            output_dict["predictions"] = torch.argmax(logits, dim=1)

            # Accumulate Top-1 accuracy for current batch
            self.top1(logits, labels)

        return output_dict

    def get_metric(self, reset: bool = True):
        r"""Return accumulated metric after validation."""
        return {f"{self.layer_name}_top1": self.top1.get_metric(reset=reset)}
