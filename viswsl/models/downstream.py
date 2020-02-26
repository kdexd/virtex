from typing import Any, Dict, List

import torch
from torch import nn

from viswsl.utils.metrics import ImageNetTopkAccuracy


class FeatureExtractor9k(nn.Module):
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
    layer_names: list, optional (default = ["layer4"])
        Which layers from ResNet to extract features from. List must contain
        a subset of ``{"layer1", "layer2", "layer3", "layer4"}``.
    normalize_with_bn: bool, optional (default = False)
        Whether to use :class:`~torch.nn.BatchNorm2d` for normalizing features.
        If this is ``False``, we flatten the features and do :func:`torch.norm`.
        This flag is ``True`` for ImageNet linear classification protocol and
        ``False`` for VOC07 linear SVM classification.
    """

    def __init__(
        self,
        trained_model: nn.Module,
        layer_names: List[str] = ["layer4"],
        normalize_with_bn: bool = False,
    ):
        super().__init__()
        self.visual: nn.Module = trained_model.visual  # type: ignore
        self.normalize_with_bn = normalize_with_bn

        # Check if layer names are all valid.
        for layer_name in layer_names:
            if layer_name not in {"layer1", "layer2", "layer3", "layer4"}:
                raise ValueError(f"Invalid layer name: {layer_name}")

        # These pooling layers will downsample features from ResNet-like models
        # so their size is ~9000 when flattened.
        self._layer1_pool = nn.AdaptiveAvgPool2d(6)  # 256 channels
        self._layer2_pool = nn.AdaptiveAvgPool2d(4)  # 512 channels
        self._layer3_pool = nn.AdaptiveAvgPool2d(3)  # 1024 channels
        self._layer4_pool = nn.AdaptiveAvgPool2d(2)  # 2048 channels

        # BatchNorm layers for normalizing features channel-wise for ImageNet
        # linear classification protocol. These do not have trainable weights.
        self._bn1 = nn.BatchNorm2d(256, affine=False, eps=1e-5, momentum=0.1)
        self._bn2 = nn.BatchNorm2d(512, affine=False, eps=1e-5, momentum=0.1)
        self._bn3 = nn.BatchNorm2d(1024, affine=False, eps=1e-5, momentum=0.1)
        self._bn4 = nn.BatchNorm2d(2048, affine=False, eps=1e-5, momentum=0.1)

        # fmt: off
        # A dict of references to layers for convenience.
        self.pool = {
            "layer1": self._layer1_pool, "layer2": self._layer2_pool,
            "layer3": self._layer3_pool, "layer4": self._layer4_pool,
        }
        self.bn = {
            "layer1": self._bn1, "layer2": self._bn2,
            "layer3": self._bn3, "layer4": self._bn4,
        }
        self.layer_names = layer_names
        # fmt: on

    @property
    def feature_size(self):
        r"""
        Output feature size (flattened) from each layer. Useful for instantiating
        linear layers on top of this module.
        """
        return {"layer1": 9216, "layer2": 8192, "layer3": 9216, "layer4": 8192}

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:

        features = self.visual(images, return_intermediate_outputs=True)

        # keys: {"layer1", "layer2", "layer3", "layer4"}
        for layer_name in features:
            if layer_name in self.layer_names:
                pooled = self.pool[layer_name](features[layer_name])

                # Perform normalization and flattening of features.
                if self.normalize_with_bn:
                    pooled = self.bn[layer_name](pooled)
                    pooled = pooled.view(pooled.size(0), -1)
                else:
                    pooled = pooled.view(pooled.size(0), -1)
                    pooled = pooled / torch.norm(pooled, dim=-1).unsqueeze(-1)

                features[layer_name] = pooled

        return features


class LinearClassifiers(nn.Module):
    r"""
    Simple linear layers for linear classification protocol on ImageNet and
    Places205 datasets. This module does K-way (K = 1000 or 205) classification
    on input images. Currently only supports training off of ``layer3`` (or
    ``res4``) and ``layer4`` (or ``res5``) stages of ResNet.

    This module can compute cross entropy loss and accumulate Top-1 accuracy
    during validation.

    Parameters
    ----------
    feature_extractor: FeatureExtractor9k
        Feature extractor on top of which we need to initialize linear layers.
    num_classes: int, optional (default = 1000)
        Number of output classes (for softmax). Set to 1000 for ImageNet and
        205 for Places205.
    """

    def __init__(
        self, feature_extractor: FeatureExtractor9k, num_classes: int = 1000
    ):
        super().__init__()
        # DO NOT set `feature_extractor` as a module in this class, else it
        # messes with the weight decay duing optimization.

        # Two linear classifiers for last two stages.
        self.layer3_fc = nn.Linear(
            feature_extractor.feature_size["layer3"], num_classes
        )
        self.layer4_fc = nn.Linear(
            feature_extractor.feature_size["layer4"], num_classes
        )
        # Initialize weights from N(0.0, 0.01) - following evaluation protocol.
        self.layer3_fc.weight.data.normal_(mean=0.0, std=0.01)
        self.layer4_fc.weight.data.normal_(mean=0.0, std=0.01)

        self.loss = nn.CrossEntropyLoss()
        self.layer3_top1 = ImageNetTopkAccuracy(top_k=1)
        self.layer4_top1 = ImageNetTopkAccuracy(top_k=1)

    def forward(
        self, features: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, num_classes)
        layer3_logits = self.layer3_fc(features["layer3"])
        layer4_logits = self.layer4_fc(features["layer4"])

        # Calculate loss if `labels` provided (not provided during inference).
        output_dict: Dict[str, Any] = {
            "loss": {
                "layer3": self.loss(layer3_logits, labels),
                "layer4": self.loss(layer4_logits, labels),
            }
        }
        # Output predictions in inference mode (no provided `labels`).
        if not self.training:
            # shape: (batch_size, )
            output_dict["layer3_predictions"] = torch.argmax(layer3_logits, dim=1)
            output_dict["layer4_predictions"] = torch.argmax(layer4_logits, dim=1)

            self.layer3_top1(labels, layer3_logits)
            self.layer4_top1(labels, layer4_logits)
        return output_dict

    def get_metric(self, reset: bool = True):
        r"""Return accumulated metric after validation."""
        return {
            "layer3_top1": self.layer3_top1.get_metric(reset=reset),
            "layer4_top1": self.layer4_top1.get_metric(reset=reset),
        }
