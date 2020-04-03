from typing import Any, Dict

import torch
from torch import nn

from viswsl.utils.metrics import TopkAccuracy


class FeatureExtractor(nn.Module):
    r"""
    Extract features from intermediate ResNet layers (stages) pool them such
    that their feature dimension is approximately 9000 (except 2048-d global
    average pooled features). These features can be used to train SVMs / linear
    layers to evaluate quality of learned representations.

    Features extraction and pooling is consistent with prior works like FAIR
    Self Supervision Benchmark `(Goyal et al, 2019) <https://arxiv.org/abs/1905.01235>`_,
    and Split-Brain Autoencoder `(Zhang et al, 2016b) <https://arxiv.org/abs/1611.09842>`_.

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


class LinearClassifier(nn.Module):
    r"""
    A simple linear layer for classification of features from a frozen feature
    extractor. This is used for ImageNet linear classification. This module
    computes cross entropy loss during forward pass, and also accumulates Top-1
    accuracy during validation.

    Parameters
    ----------
    feature_size: int, optional (default = 2048)
        Size of the last dimension of input features. Default 2048, size of
        global average pooled features from ResNet-50.
    num_classes: int, optional (default = 1000)
        Number of output classes (for softmax). Default 1000 for ImageNet.
    """

    def __init__(self, feature_size: int = 2048, num_classes: int = 1000):
        super().__init__()
        self.fc = nn.Linear(feature_size, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.top1 = TopkAccuracy(top_k=1)

        torch.nn.init.normal_(self.fc.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Any]:

        # shape: (batch_size, num_classes)
        batch_size = features.size(0)

        features = features.view(batch_size, -1)
        logits = self.fc(features)

        loss = self.loss(logits, labels)

        # Calculate cross entropy loss using `labels`.
        output_dict: Dict[str, Any] = {"loss": loss}

        if not self.training:
            # shape: (batch_size, )
            output_dict["predictions"] = torch.argmax(logits, dim=1)

            # Accumulate Top-1 accuracy for current batch
            self.top1(logits, labels)

        return output_dict

    def get_metric(self, reset: bool = True):
        r"""Return accumulated metric after validation."""
        return {f"top1": self.top1.get_metric(reset=reset)}
