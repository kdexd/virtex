r"""
A utility module which provides functionality to easily load common VirTex
models (optionally with pretrained weights) using a single line of code.

Get our full best performing VirTex model (with pretrained weights as):

>>> import virtex.model_zoo as mz
>>> model = mz.get("width_ablations/bicaptioning_R_50_L1_H2048.yaml", pretrained=True)

Any config available in ``configs/`` directory under project root can be
specified here, although this command need not be executed from project root.

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
        "task_ablations/bicaptioning_R_50_L1_H2048.yaml": "fm1nq819q74vr0kqcd3gkivlzf06xvko.pth",
        "task_ablations/captioning_R_50_L1_H2048.yaml": "7fopt8k2eutz9qvth2hh6j00o7z4o7ps.pth",
        "task_ablations/token_classification_R_50.yaml": "qwvfnji51g4gvba7i5mrw2ph5z8yfty9.pth",
        "task_ablations/multilabel_classification_R_50.yaml": "tk1hlcue9c3268bds3h036ckk7a9btlr.pth",

        # Width Ablations
        "width_ablations/bicaptioning_R_50_L1_H512.yaml": "qostt3be0pgnd0xf55vdte3wa49x6k99.pth",
        "width_ablations/bicaptioning_R_50_L1_H768.yaml": "v0p80tya0wjgsj0liqyvt386903xbwxc.pth",
        "width_ablations/bicaptioning_R_50_L1_H1024.yaml": "s2o3tvujcx2djoz1ouvuea27hrys1fbm.pth",
        "width_ablations/bicaptioning_R_50_L1_H2048.yaml": "fm1nq819q74vr0kqcd3gkivlzf06xvko.pth",

        # Depth Ablations
        "depth_ablations/bicaptioning_R_50_L1_H1024.yaml": "s2o3tvujcx2djoz1ouvuea27hrys1fbm.pth",
        "depth_ablations/bicaptioning_R_50_L2_H1024.yaml": "5enura2ao2b0iyigcuikfsdd0osun0it.pth",
        "depth_ablations/bicaptioning_R_50_L3_H1024.yaml": "xit11ev6h3q7h8wth5qokewxcn6yot2n.pth",
        "depth_ablations/bicaptioning_R_50_L4_H1024.yaml": "secpwhjx9oq59mkzsztjaews6n3680bj.pth",

        # Backbone Ablations
        "backbone_ablations/bicaptioning_R_50_L1_H1024.yaml": "s2o3tvujcx2djoz1ouvuea27hrys1fbm.pth",
        "backbone_ablations/bicaptioning_R_50W2X_L1_H1024.yaml": "0rlu15xq796tz3ebvz7lf5dbpti421le.pth",
        "backbone_ablations/bicaptioning_R_101_L1_H1024.yaml": "i3p45pr78jdz74r29qkj23v8kzb6gcsq.pth",
    }
    # Backbone from best model: fotpti1uk6bpoobeazysfc6fdbndvy90.pth


def get(config_path, pretrained: bool = False):
    r"""
    Get a model specified by relative path under Detectron2's official ``configs/`` directory.

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
