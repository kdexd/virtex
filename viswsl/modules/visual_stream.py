from typing import Dict, Union

import torch
from torch import nn
from torchvision import models as tv_models


class TorchvisionVisualStream(nn.Module):
    def __init__(self, name: str, pretrained: bool = False, **kwargs):
        super().__init__()
        self.visual_feature_size = 2048

        try:
            model_creation_method = getattr(tv_models, name)
        except AttributeError as err:
            raise AttributeError(f"{name} if not a torchvision model.")

        self._cnn = model_creation_method(
            pretrained, zero_init_residual=True, **kwargs
        )

        # Do nothing after the final residual stage.	
        self._cnn.avgpool = nn.Identity()	
        self._cnn.fc = nn.Identity()	

        # Keep a list of intermediate layer names.
        self._stage_names = [f"layer{i}" for i in range(1, 5)]

    def forward(
        self, image: torch.Tensor, return_intermediate_outputs: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        # Iterate through the modules in sequence and collect feature
        # vectors for last layers in each stage.
        intermediate_outputs: Dict[str, torch.Tensor] = {}
        for idx, (name, layer) in enumerate(self._cnn.named_children()):
            out = layer(image) if idx == 0 else layer(out)
            if name in self._stage_names:
                intermediate_outputs[name] = out

        if return_intermediate_outputs:
            return intermediate_outputs
        else:
            # shape: (batch_size, feature_size, ...)
            return intermediate_outputs["layer4"]

    def detectron2_backbone_state_dict(self):
        r"""
        Return state dict for loading as a backbone in detectron2. Useful for
        object detection downstream tasks.
        """
        # Detectron2 backbones have slightly different module names, this mapping
        # lists substrings of module names required to be renamed for loading a
        # torchvision model into Detectron2.
        DETECTRON2_RENAME_MAPPING: Dict[str, str] = {
            "layer1": "res2",
            "layer2": "res3",
            "layer3": "res4",
            "layer4": "res5",
            "bn1": "conv1.norm",
            "bn2": "conv2.norm",
            "bn3": "conv3.norm",
            "downsample.0": "shortcut",
            "downsample.1": "shortcut.norm",
        }
        # Populate this dict by renaming module names.
        d2_backbone_dict: Dict[str, torch.Tensor] = {}

        for name, param in self._cnn.state_dict().items():
            for old, new in DETECTRON2_RENAME_MAPPING.items():
                name = name.replace(old, new)

            # First conv and bn module parameters are prefixed with "stem.".
            if not name.startswith("res"):
                name = f"stem.{name}"

            d2_backbone_dict[name] = param

        return d2_backbone_dict


class BlindVisualStream(nn.Module):
    r"""A visual stream which cannot see the image."""

    def __init__(self, bias: torch.Tensor = torch.ones(49, 2048)):
        super().__init__()
        self.visual_feature_size = 2048

        # We never update the bias because a blind model cannot learn anything
        # about the image.
        self._bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, image: torch.Tensor):
        batch_size = image.size(0)
        return self._bias.unsqueeze(0).repeat(batch_size, 1, 1)
