from typing import Dict

import torch
from torch import nn


class VisualIntermediateOutputPooler(nn.Module):
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

    def __init__(self, mode: str = "avg", flatten: bool = True):
        super().__init__()

        layer = nn.AdaptiveAvgPool2d if mode == "avg" else nn.AdaptiveMaxPool2d
        self._flatten = flatten

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
        self, intermediate_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # keys: {"layer1", "layer2", "layer3", "layer4"}
        for key in intermediate_outputs:
            pooled = self._pool[key](intermediate_outputs[key])
            if self._flatten:
                pooled = pooled.view(pooled.size(0), -1)
            intermediate_outputs[key] = pooled

        return intermediate_outputs
