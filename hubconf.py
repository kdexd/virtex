dependencies = ["torch"]

import torch
import torchvision


R50_URL = "https://www.dropbox.com/s/pxgjxcva7oypf12/backbone_bicaptioning_R_50_L1_H2048.pth?dl=1"


def resnet50(pretrained: bool = False, **kwargs):
    r"""
    ResNet-50 visual backbone from the best performing VirTex model: pretrained
    for bicaptioning on COCO Captions, with textual head ``L = 1, H = 2048``.

    This is a torchvision-like model, with the last ``avgpool`` and `fc``
    modules replaced with ``nn.Identity()`` modules. Given a batch of image
    tensors with size ``(B, 3, 224, 224)``, this model computes spatial image
    features of size ``(B, 7, 7, 2048)``, where B = batch size.

    pretrained (bool): Whether to load model with pretrained weights.
    """

    # Create a torchvision resnet50 with randomly initialized weights.
    model = torchvision.models.resnet50(pretrained=False, **kwargs)

    # Replace global average pooling and fully connected layers with identity
    # modules.
    model.avgpool = torch.nn.Identity()
    model.fc = torch.nn.Identity()

    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(R50_URL, progress=False)
        )
    return model
