"""
Finetune a pre-trained model on a downstream task, one of those available in
Detectron2.
Supported downstream:
  - LVIS Instance Segmentation
  - COCO Instance Segmentation
  - Pascal VOC 2007+12 Object Detection

Reference: https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py
Thanks to the developers of Detectron2!
"""
import argparse
import os
import re
from typing import Any, Dict, Union

import torch
from torch.utils.tensorboard import SummaryWriter

import detectron2 as d2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.evaluation import (
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    COCOEvaluator,
)
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads

from virtex.config import Config
from virtex.factories import PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser
import virtex.utils.distributed as dist

# fmt: off
parser = common_parser(
    description="Train object detectors from pretrained visual backbone."
)
parser.add_argument(
    "--d2-config", required=True,
    help="Path to a detectron2 config for downstream task finetuning."
)
parser.add_argument(
    "--d2-config-override", nargs="*", default=[],
    help="""Key-value pairs from Detectron2 config to override from file.
    Some keys will be ignored because they are set from other args:
    [DATALOADER.NUM_WORKERS, SOLVER.EVAL_PERIOD, SOLVER.CHECKPOINT_PERIOD,
    TEST.EVAL_PERIOD, OUTPUT_DIR]""",
)

parser.add_argument_group("Checkpointing and Logging")
parser.add_argument(
    "--weight-init", choices=["random", "imagenet", "torchvision", "virtex"],
    default="virtex", help="""How to initialize weights:
        1. 'random' initializes all weights randomly
        2. 'imagenet' initializes backbone weights from torchvision model zoo
        3. {'torchvision', 'virtex'} load state dict from --checkpoint-path
            - with 'torchvision', state dict would be from PyTorch's training
              script.
            - with 'virtex' it should be for our full pretrained model."""
)
parser.add_argument(
    "--checkpoint-path",
    help="Path to load checkpoint and run downstream task evaluation."
)
parser.add_argument(
    "--resume", action="store_true", help="""Specify this flag when resuming
    training from a checkpoint saved by Detectron2."""
)
parser.add_argument(
    "--eval-only", action="store_true",
    help="Skip training and evaluate checkpoint provided at --checkpoint-path.",
)
parser.add_argument(
    "--checkpoint-every", type=int, default=5000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
# fmt: on


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    r"""
    ROI head with ``res5`` stage followed by a BN layer. Used with Faster R-CNN
    C4/DC5 backbones for VOC detection.
    """

    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = d2.layers.get_norm(cfg.MODEL.RESNETS.NORM, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


def build_detectron2_config(_C: Config, _A: argparse.Namespace):
    r"""Build detectron2 config based on our pre-training config and args."""
    _D2C = d2.config.get_cfg()

    # Override some default values based on our config file.
    _D2C.merge_from_file(_A.d2_config)
    _D2C.merge_from_list(_A.d2_config_override)

    # Set some config parameters from args.
    _D2C.DATALOADER.NUM_WORKERS = _A.cpu_workers
    _D2C.SOLVER.CHECKPOINT_PERIOD = _A.checkpoint_every
    _D2C.OUTPUT_DIR = _A.serialization_dir

    # Set ResNet depth to override in Detectron2's config.
    _D2C.MODEL.RESNETS.DEPTH = int(
        re.search(r"resnet(\d+)", _C.MODEL.VISUAL.NAME).group(1)
        if "torchvision" in _C.MODEL.VISUAL.NAME
        else re.search(r"_R_(\d+)", _C.MODEL.VISUAL.NAME).group(1)
        if "detectron2" in _C.MODEL.VISUAL.NAME
        else 0
    )
    return _D2C


class DownstreamTrainer(DefaultTrainer):
    r"""
    Extension of detectron2's ``DefaultTrainer``: custom evaluator and hooks.

    Parameters
    ----------
    cfg: detectron2.config.CfgNode
        Detectron2 config object containing all config params.
    weights: Union[str, Dict[str, Any]]
        Weights to load in the initialized model. If ``str``, then we assume path
        to a checkpoint, or if a ``dict``, we assume a state dict. This will be
        an ``str`` only if we resume training from a Detectron2 checkpoint.
    """

    def __init__(self, cfg, weights: Union[str, Dict[str, Any]]):

        super().__init__(cfg)

        # Load pre-trained weights before wrapping to DDP because `ApexDDP` has
        # some weird issue with `DetectionCheckpointer`.
        # fmt: off
        if isinstance(weights, str):
            # weights are ``str`` means ImageNet init or resume training.
            self.start_iter = (
                DetectionCheckpointer(
                    self._trainer.model,
                    optimizer=self._trainer.optimizer,
                    scheduler=self.scheduler
                ).resume_or_load(weights, resume=True).get("iteration", -1) + 1
            )
        elif isinstance(weights, dict):
            # weights are a state dict means our pretrain init.
            DetectionCheckpointer(self._trainer.model)._load_model(weights)
        # fmt: on

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = d2.data.MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "coco":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)

    def test(self, cfg=None, model=None, evaluators=None):
        r"""Evaluate the model and log results to stdout and tensorboard."""
        cfg = cfg or self.cfg
        model = model or self.model

        tensorboard_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
        results = super().test(cfg, model)
        flat_results = d2.evaluation.testing.flatten_results_dict(results)
        for k, v in flat_results.items():
            tensorboard_writer.add_scalar(k, v, self.start_iter)


def main(_A: argparse.Namespace):

    # Get the current device as set for current distributed process.
    # Check `launch` function in `virtex.utils.distributed` module.
    device = torch.cuda.current_device()

    # Local process group is needed for detectron2.
    pg = list(range(dist.get_world_size()))
    d2.utils.comm._LOCAL_PROCESS_GROUP = torch.distributed.new_group(pg)

    # Create a config object (this will be immutable) and perform common setup
    # such as logging and setting up serialization directory.
    if _A.weight_init == "imagenet":
        _A.config_override.extend(["MODEL.VISUAL.PRETRAINED", True])
    _C = Config(_A.config, _A.config_override)

    # We use `default_setup` from detectron2 to do some common setup, such as
    # logging, setting up serialization etc. For more info, look into source.
    _D2C = build_detectron2_config(_C, _A)
    default_setup(_D2C, _A)

    # Prepare weights to pass in instantiation call of trainer.
    if _A.weight_init in {"virtex", "torchvision"}:
        if _A.resume:
            # If resuming training, let detectron2 load weights by providing path.
            model = None
            weights = _A.checkpoint_path
        else:
            # Load backbone weights from VirTex pretrained checkpoint.
            model = PretrainingModelFactory.from_config(_C)
            if _A.weight_init == "virtex":
                CheckpointManager(model=model).load(_A.checkpoint_path)
            else:
                model.visual.cnn.load_state_dict(
                    torch.load(_A.checkpoint_path, map_location="cpu")["state_dict"],
                    strict=False,
                )
            weights = model.visual.detectron2_backbone_state_dict()
    else:
        # If random or imagenet init, just load weights after initializing model.
        model = PretrainingModelFactory.from_config(_C)
        weights = model.visual.detectron2_backbone_state_dict()

    # Back up pretrain config and model checkpoint (if provided).
    _C.dump(os.path.join(_A.serialization_dir, "pretrain_config.yaml"))
    if _A.weight_init == "virtex" and not _A.resume:
        torch.save(
            model.state_dict(),
            os.path.join(_A.serialization_dir, "pretrain_model.pth"),
        )

    del model
    trainer = DownstreamTrainer(_D2C, weights)
    trainer.test() if _A.eval_only else trainer.train()


if __name__ == "__main__":
    _A = parser.parse_args()

    # This will launch `main` and set appropriate CUDA device (GPU ID) as
    # per process (accessed in the beginning of `main`).
    dist.launch(
        main,
        num_machines=_A.num_machines,
        num_gpus_per_machine=_A.num_gpus_per_machine,
        machine_rank=_A.machine_rank,
        dist_url=_A.dist_url,
        args=(_A, ),
    )
