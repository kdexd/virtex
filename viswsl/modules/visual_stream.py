import torch
from torch import nn
from torchvision.models import resnext50_32x4d


class VisualStream(nn.Module):
    # It's just a ResNeXt here right now. Making a separate class to make it
    # more configurable.

    def __init__(self):
        super().__init__()

        self._cnn = resnext50_32x4d(pretrained=False)

        # Set the global average pooling and FC layers as identity.
        # self._cnn.avgpool = nn.Identity()
        self._cnn.fc = nn.Identity()

    def forward(self, image: torch.Tensor):
        # Get a flat feature vector, view it as spatial features.
        # TODO (kd): Hardcoded values now, deal with them later.
        # shape: (batch_size, 7 * 7 * 2048)
        flat_spatial_features = self._cnn(image)

        # shape: (batch_size, 7, 7, 2048)
        # spatial_features = flat_spatial_features.view(-1, 7, 7, 2048)
        return flat_spatial_features
