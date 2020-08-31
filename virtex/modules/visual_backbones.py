from typing import Any, Dict, Union

import torch
from torch import nn
import torchvision


class VisualBackbone(nn.Module):
    r"""
    Base class for all visual backbones. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.
    """

    def __init__(self, visual_feature_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size


class BlindVisualBackbone(VisualBackbone):
    r"""
    A visual backbone which cannot see the image. It always outputs a tensor
    filled with constant value.

    Parameters
    ----------
    visual_feature_size: int, optional (default = 2048)
        Size of the last dimension (channels) of output from forward pass.
    bias_value: float, optional (default = 1.0)
        Constant value to fill in the output tensor.
    """

    def __init__(self, visual_feature_size: int = 2048, bias_value: float = 1.0):
        super().__init__(visual_feature_size)

        # We never update the bias because a blind model cannot learn anything
        # about the image. Add an axis for proper broadcasting.
        self._bias = nn.Parameter(
            torch.full((1, self.visual_feature_size), fill_value=bias_value),
            requires_grad=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""
        Compute visual features for a batch of input images. Since this model
        is *blind*, output will always be constant.

        Parameters
        ----------
        image: torch.Tensor
            Batch of input images. A tensor of shape
            ``(batch_size, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Output visual features, filled with :attr:`bias_value`. A tensor of
            shape ``(batch_size, visual_feature_size)``.
        """
        batch_size = image.size(0)
        return self._bias.repeat(batch_size, 1)


class TorchvisionVisualBackbone(VisualBackbone):
    r"""
    A visual backbone from `Torchvision model zoo
    <https://pytorch.org/docs/stable/torchvision/models.html>`_. Any model can
    be specified using corresponding method name from the model zoo.

    Parameters
    ----------
    name: str, optional (default = "resnet50")
        Name of the model from Torchvision model zoo.
    visual_feature_size: int, optional (default = 2048)
        Size of the channel dimension of output visual features from forward pass.
    pretrained: bool, optional (default = False)
        Whether to load ImageNet pretrained weights from Torchvision.
    frozen: float, optional (default = False)
        Whether to keep all weights frozen during training.
    """

    def __init__(
        self,
        name: str = "resnet50",
        visual_feature_size: int = 2048,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__(visual_feature_size)

        self.cnn = getattr(torchvision.models, name)(
            pretrained, zero_init_residual=True
        )
        # Do nothing after the final residual stage.
        self.cnn.fc = nn.Identity()

        # Freeze all weights if specified.
        if frozen:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval()

        # Keep a list of intermediate layer names.
        self._stage_names = [f"layer{i}" for i in range(1, 5)]

    def forward(
        self, image: torch.Tensor, return_intermediate_outputs: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Compute visual features for a batch of input images.

        Parameters
        ----------
        image: torch.Tensor
            Batch of input images. A tensor of shape
            ``(batch_size, 3, height, width)``.
        return_intermediate_outputs: bool, optional (default = False)
            Whether to return feaures extracted from all intermediate stages or
            just the last one. This can only be set ``True`` when using a
            ResNet-like model.

        Returns
        -------
        Union[torch.Tensor, Dict[str, torch.Tensor]]
            - If ``return_intermediate_outputs = False``, this will be a tensor
              of shape ``(batch_size, channels, height, width)``, for example
              it will be ``(batch_size, 2048, 7, 7)`` for ResNet-50 (``layer4``).

            - If ``return_intermediate_outputs = True``, this will be a dict
              with keys ``{"layer1", "layer2", "layer3", "layer4", "avgpool"}``
              containing features from all intermediate layers and global
              average pooling layer.
        """

        # Iterate through the modules in sequence and collect feature
        # vectors for last layers in each stage.
        intermediate_outputs: Dict[str, torch.Tensor] = {}
        for idx, (name, layer) in enumerate(self.cnn.named_children()):
            out = layer(image) if idx == 0 else layer(out)
            if name in self._stage_names:
                intermediate_outputs[name] = out

        # Add pooled spatial features.
        intermediate_outputs["avgpool"] = torch.mean(
            intermediate_outputs["layer4"], dim=[2, 3]
        )
        if return_intermediate_outputs:
            return intermediate_outputs
        else:
            # shape: (batch_size, channels, height, width)
            return intermediate_outputs["layer4"]

    def detectron2_backbone_state_dict(self) -> Dict[str, Any]:
        r"""
        Return state dict of visual backbone which can be loaded with
        `Detectron2 <https://github.com/facebookresearch/detectron2>`_.
        This is useful for downstream tasks based on Detectron2 (such as
        object detection and instance segmentation). This method renames
        certain parameters from Torchvision-style to Detectron2-style.

        Returns
        -------
        Dict[str, Any]
            A dict with three keys: ``{"model", "author", "matching_heuristics"}``.
            These are necessary keys for loading this state dict properly with
            Detectron2.
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

        for name, param in self.cnn.state_dict().items():
            for old, new in DETECTRON2_RENAME_MAPPING.items():
                name = name.replace(old, new)

            # First conv and bn module parameters are prefixed with "stem.".
            if not name.startswith("res"):
                name = f"stem.{name}"

            d2_backbone_dict[name] = param

        return {
            "model": d2_backbone_dict,
            "__author__": "Karan Desai",
            "matching_heuristics": True,
        }
