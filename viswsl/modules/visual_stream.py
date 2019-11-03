import torch
from torch import nn
from torchvision import models as tv_models


class TorchvisionVisualStream(nn.Module):
    def __init__(self, name: str, pretrained: bool = False, **kwargs):
        super().__init__()
        try:
            model_creation_method = getattr(tv_models, name)
        except AttributeError as err:
            raise AttributeError(f"{name} if not a torchvision model.")

        self._cnn = model_creation_method(pretrained, **kwargs)

        # Do nothing after the global average pooling layer.
        self._cnn.fc = nn.Identity()

    def forward(self, image: torch.Tensor):
        # Get a flat feature vector, view it as spatial features.
        # TODO (kd): Hardcoded values now, deal with them later.
        # shape: (batch_size, 7 * 7 * 2048)
        flat_spatial_features = self._cnn(image)

        # shape: (batch_size, 7, 7, 2048)
        # spatial_features = flat_spatial_features.view(-1, 7, 7, 2048)
        return flat_spatial_features


class BlindVisualStream(nn.Module):
    r"""A visual stream which cannot see the image."""

    def __init__(self, bias: torch.Tensor = torch.ones(2048)):
        super().__init__()

        # We never update the bias because a blind model cannot learn anything
        # about the image.
        self._bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, image: torch.Tensor):
        batch_size = image.size(0)
        return self._bias.unsqueeze(0).repeat(batch_size, 1)
