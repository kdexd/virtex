from typing import Dict, List

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
    """

    def __init__(
        self, trained_model: nn.Module, layer_names: List[str] = ["layer4"]
    ):
        super().__init__()
        self.visual: nn.Module = trained_model.visual  # type: ignore

        # Check if layer names are all valid.
        for layer_name in layer_names:
            if layer_name not in {"layer1", "layer2", "layer3", "layer4"}:
                raise ValueError(f"Invalid layer name: {layer_name}")

        # These pooling layers will downsample features from ResNet-like models
        # so their size is ~9000 when flattened.
        self._layer1_pool = nn.AdaptiveAvgPool2d(6)
        self._layer2_pool = nn.AdaptiveAvgPool2d(4)
        self._layer3_pool = nn.AdaptiveAvgPool2d(3)
        self._layer4_pool = nn.AdaptiveAvgPool2d(2)

        # fmt: off
        # A dict of references to layers for convenience.
        self.pool = {
            "layer1": self._layer1_pool, "layer2": self._layer2_pool,
            "layer3": self._layer3_pool, "layer4": self._layer4_pool,
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

                # Flatten and normalize features.
                pooled = pooled.view(pooled.size(0), -1)
                pooled = pooled / torch.norm(pooled, dim=-1).unsqueeze(-1)
                features[layer_name] = pooled

        return features


class ImageNetLinearClassifier(nn.Module):
    r"""
    A simple linear layer for ImageNet linear classification protocol, performs
    1000-way classification on input images. Currently only supports training
    off of last stage of ResNet (``layer4`` in torchvision naming, ``res5`` in
    MSRA or Caffe2 naming).

    Note
    ----
    This class is initialized with the whole pre-trained model, but its
    ``state_dict`` will only contain the linear layer because backbone is frozen
    and never updated.
    """

    def __init__(self, trained_model: nn.Module):
        super().__init__()

        # Only get features from last layer of backbone.
        self.feature_extractor = FeatureExtractor9k(trained_model, ["layer4"])

        # Freeze the backbone completely.
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Linear classifier on top of the backbone.
        self.fc = nn.Linear(self.feature_extractor.feature_size["layer4"], 1000)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy_accumulator = ImageNetTopkAccuracy(top_k=1)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # Extracted features of size ~9000, flattened and normalized.
        with torch.no_grad():
            features = self.feature_extractor(images)["layer4"]

        # shape: (batch_size, 1000)
        logits = self.fc(features)

        # Calculate loss if `labels` provided (not provided during inference).
        output_dict = {"loss": self.loss(logits, labels)}

        # Output predictions in inference mode (no provided `labels`).
        if not self.training:
            # shape: (batch_size, )
            output_dict["predictions"] = torch.argmax(logits, dim=1)
            self.accuracy_accumulator(labels, logits)
        return output_dict

    def state_dict(self):
        r"""
        Override super method to only include weights from linear layer,
        because the backbone is frozen.
        """
        return self.fc.state_dict()

    def load_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ):
        r"""Override super method to match with :meth:`state_dict`."""
        self.fc.load_state_dict(state_dict, strict=strict)

    def get_metric(self, reset: bool = True):
        r"""Return accumulated metric after validation."""
        return self.accuracy_accumulator.get_metric(reset=reset)
