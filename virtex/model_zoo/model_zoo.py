r"""
A utility module to easily load common VirTex models (optionally with pretrained
weights) using a single line of code.

Get our full best performing VirTex model (with pretrained weights as):

>>> import virtex.model_zoo as mz
>>> model = mz.get("width_ablations/bicaptioning_R_50_L1_H2048.yaml", pretrained=True)

Any config available in ``configs/`` directory under project root can be
specified here, although this command need not be executed from project root.
For more details on available models, refer :doc:`usage/model_zoo`.

Part of this code is adapted from Detectron2's model zoo; which was originally
implemented by the developers of this codebase, with reviews and further
changes by Detectron2 developers.
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pkg_resources

from fvcore.common.download import download
import torch

from virtex.config import Config
from virtex.factories import PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager


class _ModelZooUrls(object):
    r"""Mapping from config names to URL suffixes of pretrained weights."""

    URL_PREFIX = "https://www.dropbox.com/s"

    CONFIG_PATH_TO_DB_ID = {

        # Pretraining Task Ablations
        "task_ablations/bicaptioning_R_50_L1_H2048.yaml": "mbeeso8wyieq8wy",
        "task_ablations/captioning_R_50_L1_H2048.yaml": "r6zen9k43m5oo58",
        "task_ablations/token_classification_R_50.yaml": "o4p9lki505r0mef",
        "task_ablations/multilabel_classification_R_50.yaml": "hbspp3jv3u8h3bc",
        "task_ablations/masked_lm_R_50_L1_H2048.yaml": "ldzrk6vem4mg6bl",

        # Width Ablations
        "width_ablations/bicaptioning_R_50_L1_H512.yaml": "o9fr69jjqfn8a65",
        "width_ablations/bicaptioning_R_50_L1_H768.yaml": "1zxglqrrbfufv9d",
        "width_ablations/bicaptioning_R_50_L1_H1024.yaml": "pdat4tvhnqxel64",
        "width_ablations/bicaptioning_R_50_L1_H2048.yaml": "mbeeso8wyieq8wy",

        # Depth Ablations
        "depth_ablations/bicaptioning_R_50_L1_H1024.yaml": "pdat4tvhnqxel64",
        "depth_ablations/bicaptioning_R_50_L2_H1024.yaml": "ft1vtt4okirzjgo",
        "depth_ablations/bicaptioning_R_50_L3_H1024.yaml": "5ldo1rcsnrshmjr",
        "depth_ablations/bicaptioning_R_50_L4_H1024.yaml": "zgiit2wcluuq3xh",

        # Backbone Ablations
        "backbone_ablations/bicaptioning_R_50_L1_H1024.yaml": "pdat4tvhnqxel64",
        "backbone_ablations/bicaptioning_R_50W2X_L1_H1024.yaml": "5o198ux709r6376",
        "backbone_ablations/bicaptioning_R_101_L1_H1024.yaml": "bb74jubt68cpn80",
    }


def get(config_path, pretrained: bool = False):
    r"""
    Get a model specified by relative path under Detectron2's official
    ``configs/`` directory.

    Parameters
    ----------
    config_path: str
        Name of config file relative to ``configs/`` directory under project
        root. (For example, ``width_ablations/bicaptioning_R_50_L1_H2048.yaml``)
    pretrained: bool, optional (default = False)
        If ``True``, will initialize the model with the pretrained weights. If
        ``False``, the weights will be initialized randomly.
    """

    # Get the original path to config file (shipped with inside the package).
    _pkg_config_path = pkg_resources.resource_filename(
        "virtex.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(_pkg_config_path):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))

    _C = Config(_pkg_config_path)
    model = PretrainingModelFactory.from_config(_C)

    if pretrained:
        # Get URL for the checkpoint for this config path.
        if config_path in _ModelZooUrls.CONFIG_PATH_TO_DB_ID:

            dropbox_id = _ModelZooUrls.CONFIG_PATH_TO_DB_ID[config_path]
            filename = os.path.basename(config_path).replace(".yaml", ".pth")

            checkpoint_url = f"{_ModelZooUrls.URL_PREFIX}/{dropbox_id}/{filename}?dl=1"
        else:
            raise RuntimeError("{} not available in Model Zoo!".format(config_path))

        # Download the pretrained model weights and save with a sensible name.
        # This will be downloaded only if it does not exist.
        checkpoint_path = download(
            checkpoint_url,
            dir=os.path.expanduser("~/.torch/virtex_cache"),
            filename=os.path.basename(config_path).replace(".yaml", ".pth")
        )
        CheckpointManager(model=model).load(checkpoint_path)

    return model
