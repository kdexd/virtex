import torch
from torch import nn


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
