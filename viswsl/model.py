from typing import Dict, List

import torch
from torch import nn

# TODO (kd): have attention/fusion technique as a dependency injection.
from viswsl.modules.attention import ScaledDotProductAttention


class ViswslModel(nn.Module):
    # TODO (kd): Find a better name maybe?

    def __init__(self, visual, linguistic):
        super().__init__()
        self._visual = visual
        self._linguistic = linguistic

        # TODO (kd): Remove hardcoded values once this becomes a dependency
        # injection.
        self._attention = ScaledDotProductAttention(2048, linguistic.hidden_size)
        self._linear = nn.Linear(
            2048 + linguistic.hidden_size, linguistic.vocab_size
        )
        self._loss = nn.CrossEntropyLoss(ignore_index=linguistic.padding_idx)

    def forward(
        self,
        image: torch.Tensor,
        caption_tokens: torch.Tensor,
        masked_labels: torch.Tensor,
    ):
        # shape: (batch_size, 2048, 7, 7)
        image_features = self._visual(image)

        # shape: (batch_size, 49, 2048)
        image_features = image_features.view(-1, 2048, 49).permute(0, 2, 1)

        # shape: (batch_size, max_caption_length, hidden_size)
        output_hidden = self._linguistic(caption_tokens, masked_labels)

        # shape: (batch_size, max_caption_length, 2048)
        attended_features = self._attention(image_features, output_hidden)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self._linear(
            torch.cat((attended_features, output_hidden), dim=-1)
        )

        # Get predictions from logits, only the predictions at [MASK]ed
        # positions would be useful.
        predictions = torch.argmax(output_logits, dim=-1)
        output_dict = {"predictions": predictions}

        # Collapse dimensions: convert logits to (N, C), targets to (N,).
        output_dict["loss"] = self._loss(
            output_logits.view(-1, output_logits.size(-1)), masked_labels.view(-1)
        )
        return output_dict


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

    def __init__(
        self,
        pretrained_model: ViswslModel,
        mode: str = "avg",
        normalize: bool = True,
    ):
        super().__init__()
        self._cnn = pretrained_model._visual

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
        features = self._cnn(image, return_intermediate_outputs=True)

        # keys: {"layer1", "layer2", "layer3", "layer4"}
        for layer_name in features:
            if layer_name in layer_names:
                pooled = self._pool[layer_name](features[layer_name])
                pooled = pooled.view(pooled.size(0), -1)
                if self._normalize:
                    pooled = pooled / torch.norm(pooled, dim=-1).unsqueeze(-1)
                features[layer_name] = pooled

        return features
