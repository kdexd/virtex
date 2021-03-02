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

    URL_PREFIX = "https://umich.box.com/shared/static"

    CONFIG_PATH_TO_URL_SUFFIX = {

        # Pretraining Task Ablations
        "task_ablations/bicaptioning_R_50_L1_H2048.yaml": "zu8zxtxrron29icd76owgjzojmfcgdk3.pth",
        "task_ablations/captioning_R_50_L1_H2048.yaml": "1q9qh1cj2u4r5laj7mefd2mlzwthnga7.pth",
        "task_ablations/token_classification_R_50.yaml": "idvoxjl60pzpcllkbvadqgvwazil2mis.pth",
        "task_ablations/multilabel_classification_R_50.yaml": "yvlflmo0klqy3m71p6ug06c6aeg282hy.pth",
        "task_ablations/masked_lm_R_50_L1_H2048.yaml": "x3eij00eslse9j35t9j9ijyj8zkbkizh.pth",

        # Width Ablations
        "width_ablations/bicaptioning_R_50_L1_H512.yaml": "wtk18v0vffws48u5yrj2qjt94wje1pit.pth",
        "width_ablations/bicaptioning_R_50_L1_H768.yaml": "e94n0iexdvksi252bn7sm2vqjnyt9okf.pth",
        "width_ablations/bicaptioning_R_50_L1_H1024.yaml": "1so9cu9y06gy27rqbzwvek4aakfd8opf.pth",
        "width_ablations/bicaptioning_R_50_L1_H2048.yaml": "zu8zxtxrron29icd76owgjzojmfcgdk3.pth",

        # Depth Ablations
        "depth_ablations/bicaptioning_R_50_L1_H1024.yaml": "1so9cu9y06gy27rqbzwvek4aakfd8opf.pth",
        "depth_ablations/bicaptioning_R_50_L2_H1024.yaml": "9e88f6l13a9r8wq5bbe8qnoh9zenanq3.pth",
        "depth_ablations/bicaptioning_R_50_L3_H1024.yaml": "4cv8052xiq91h7lyx52cp2a6m7m9qkgo.pth",
        "depth_ablations/bicaptioning_R_50_L4_H1024.yaml": "bk5w4471mgvwa5mv6e4c7htgsafzmfm0.pth",

        # Backbone Ablations
        "backbone_ablations/bicaptioning_R_50_L1_H1024.yaml": "1so9cu9y06gy27rqbzwvek4aakfd8opf.pth",
        "backbone_ablations/bicaptioning_R_50W2X_L1_H1024.yaml": "19vcaf1488945836kir9ebm5itgtugaw.pth",
        "backbone_ablations/bicaptioning_R_101_L1_H1024.yaml": "nptbh4jsj0c0kjsnc2hw754fkikpgx9v.pth",
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
        if config_path in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
            url_suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
            checkpoint_url = f"{_ModelZooUrls.URL_PREFIX}/{url_suffix}"
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
