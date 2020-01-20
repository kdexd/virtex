from typing import Dict, List

import torch
from torch import nn


class VOC07ClassificationFeatureExtractor(nn.Module):
    r"""
    Pool intermediate layer outputs for ResNet-like visual streams such that
    their feature size is approximately 9000. We train linear SVMs using these
    features for one vs. all classification on Pascal VOC dataset. This is
    consistent with FAIR Self Supervision Benchmark (Goyal et al, 2019).

    References
    ----------
    Scaling and Benchmarking Self-Supervised Visual Representation Learning.
    Priya Goyal, Dhruv Mahajan, Abhinav Gupta, Ishan Misra
    https://arxiv.org/abs/1905.01235
    """

    def __init__(self, trained_model, mode: str = "avg", normalize: bool = True):
        super().__init__()
        self.visual = trained_model.visual

        layer = nn.AdaptiveAvgPool2d if mode == "avg" else nn.AdaptiveMaxPool2d
        self._normalize = normalize

        # This spatial size will downsample features from ResNet-like models
        # so their size is ~9000 when flattened
        self._layer1_pool = layer(6)
        self._layer2_pool = layer(4)
        self._layer3_pool = layer(3)
        self._layer4_pool = layer(2)

        # fmt: off
        # A dict of references to layers for convenience.
        self._pool = {
            "layer1": self._layer1_pool, "layer2": self._layer2_pool,
            "layer3": self._layer3_pool, "layer4": self._layer4_pool,
        }
        # fmt: on

    def forward(
        self, image: torch.Tensor, layer_names: List[str] = None
    ) -> Dict[str, torch.Tensor]:

        layer_names = layer_names or list(self._pool.keys())
        features = self.visual(image, return_intermediate_outputs=True)

        # keys: {"layer1", "layer2", "layer3", "layer4"}
        for layer_name in features:
            if layer_name in layer_names:
                pooled = self._pool[layer_name](features[layer_name])
                pooled = pooled.view(pooled.size(0), -1)
                if self._normalize:
                    pooled = pooled / torch.norm(pooled, dim=-1).unsqueeze(-1)
                features[layer_name] = pooled

        return features
