"""
Finetune a pre-trained model on a downstream task, one of those available in
Detectron2. Supported downstream for now:
  - LVIS Instance Segmentation
  - Pascal VOC 2007+12 Object Detection

Reference: https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py
Thanks to the developers of Detectron2!
"""
import argparse
import os
import re

import torch
from torch import nn
from apex import amp

import detectron2 as d2
from detectron2.engine import SimpleTrainer, DefaultTrainer, default_setup, EvalHook
from detectron2.evaluation import LVISEvaluator, PascalVOCDetectionEvaluator

from viswsl.config import Config
from viswsl.factories import PretrainingModelFactory
import viswsl.utils.distributed as dist


parser = argparse.ArgumentParser(description="LVIS Fine-tuning with detectron2")
# fmt: off
parser.add_argument(
    "--task", required=True, choices=["lvis", "voc"],
)
parser.add_argument(
    "--config", required=True,
    help="""Path to a config file used to train the model whose checkpoint will
    be loaded (not Detectron2 config)."""
)
parser.add_argument(
    "--config-override", nargs="*", default=[],
    help="""A sequence of key-value pairs specifying certain config arguments
    (with dict-like nesting) using a dot operator.""",
)
parser.add_argument(
    "--cpu-workers", type=int, default=2, help="Number of CPU workers."
)
parser.add_argument(
    "--dist-backend", default="nccl", choices=["nccl", "gloo"],
    help="torch.distributed backend for distributed training.",
)
parser.add_argument(
    "--slurm", action="store_true",
    help="""Whether using SLURM for launching distributed training processes.
    Set `$MASTER_PORT` env variable externally for distributed process group
    communication."""
)

parser.add_argument_group("Checkpointing and Logging")
parser.add_argument(
    "--imagenet-backbone", action="store_true",
    help="""Whether to load ImageNet pre-trained weights. This flag will ignore
    weights from `--checkpoint-path`."""
)
parser.add_argument(
    "--checkpoint-path",
    help="""Path to load checkpoint and run downstream task evaluation. The
    name of checkpoint file is required to be `checkpoint_*.pth`, where * is
    iteration number from which the checkpoint was serialized."""
)
parser.add_argument(
    "--serialization-dir", default=None,
    help="""Path to a directory to save results log as a Tensorboard event
    file. If not provided, this will be the parent directory of checkpoint."""
)
parser.add_argument(
    "--checkpoint-every", type=int, default=5000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
# fmt: on


def build_detectron2_config(_C: Config, _A: argparse.Namespace):
    r"""Build detectron2 config based on our pre-training config and args."""
    _D2C = d2.config.get_cfg()

    # Override some default values based on our config file.
    if _A.task == "lvis":
        _D2C.merge_from_file(_C.DOWNSTREAM.LVIS.D2_CONFIG)
    elif _A.task == "voc":
        _D2C.merge_from_file(_C.DOWNSTREAM.VOC.D2_CONFIG)

    # Set random seed, workers etc. from args.
    _D2C.SEED = _C.RANDOM_SEED
    _D2C.DATALOADER.NUM_WORKERS = _A.cpu_workers
    _D2C.SOLVER.EVAL_PERIOD = _A.checkpoint_every
    _D2C.SOLVER.CHECKPOINT_PERIOD = _A.checkpoint_every
    _D2C.TEST.EVAL_PERIOD = _A.checkpoint_every
    _D2C.OUTPUT_DIR = _A.serialization_dir

    # Adjust learning rate and batch size by number of GPUs (linear scaling).
    # Config file has these set according to single GPU.
    _D2C.SOLVER.BASE_LR *= dist.get_world_size()
    _D2C.SOLVER.IMS_PER_BATCH *= dist.get_world_size()

    # Set ResNet depth to override in Detectron2's config.
    _D2C.MODEL.RESNETS.DEPTH = int(
        re.search(r"resnet(\d+)", _C.MODEL.VISUAL.NAME).group(1)
        if "torchvision" in _C.MODEL.VISUAL.NAME
        else re.search(r"_R_(\d+)", _C.MODEL.VISUAL.NAME).group(1)
        if "detectron2" in _C.MODEL.VISUAL.NAME
        else 0
    )
    # Override some config values if initializing by ImageNet backbone.
    if _A.imagenet_backbone:
        # Weights path, BGR format and ImageNet mean and std.
        _D2C.MODEL.WEIGHTS = (
            f"detectron2://ImageNetPretrained/MSRA/R-{_D2C.MODEL.RESNETS.DEPTH}.pkl"
        )
        _D2C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
        _D2C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
        _D2C.INPUT.FORMAT = "BGR"

    # Task-specific adjustments in config.
    if _A.task == "lvis":
        _D2C.MODEL.RESNETS.NORM = _C.DOWNSTREAM.LVIS.NORM_LAYER
        _D2C.MODEL.FPN.NORM = _C.DOWNSTREAM.LVIS.NORM_LAYER
        # If using LVIS and ImageNet backbone, use FrozenBN and no BN in FPN.
        if _A.imagenet_backbone:
            _D2C.MODEL.RESNETS.NORM = "FrozenBN"
            _D2C.MODEL.FPN.NORM = ""
    elif _A.task == "voc":
        _D2C.MODEL.RESNETS.NORM = _C.DOWNSTREAM.VOC.NORM_LAYER
        _D2C.MODEL.RESNETS.RES5_DILATION = _C.DOWNSTREAM.VOC.RES5_DILATION

    return _D2C


class LazyEvalHook(EvalHook):
    r"""Extension of detectron2's ``EvalHook``: start evaluation after few iters."""
    def __init__(self, start_after, eval_period, eval_function):
        self._start_after = start_after
        super().__init__(eval_period, eval_function)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter >= self._start_after:
            super().after_step()


class DownstreamTrainer(DefaultTrainer):
    r"""Extension of detectron2's ``DefaultTrainer``: custom evaluator and hooks."""

    def __init__(self, cfg):
        # We do not make any super call here and implement `__init__` from
        #  `DefaultTrainer`: we need to initialize mixed precision model before
        # wrapping to DDP, so we need to do it this way.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Initialize model and optimizer for mixed precision training.
        # model.backbone, optimizer = amp.initialize(
        #     model.backbone, optimizer, opt_level=f"O2"
        # )
        # Enable distributed training if we have multiple GPUs.
        if dist.get_world_size() > 1:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[dist.get_rank()], broadcast_buffers=False,
            )

        # Call `__init__` from grandparent class: `SimpleTrainer`.
        SimpleTrainer.__init__(self, model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = d2.checkpoint.DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = d2.data.MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        __C = self.cfg.clone()

        def _eval():
            # Function for ``LazyEvalHook``.
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do iteration timing, LR scheduling, checkpointing, logging etc.
        ret = [
            d2.engine.hooks.IterationTimer(),
            d2.engine.hooks.LRScheduler(self.optimizer, self.scheduler),
            d2.engine.hooks.PeriodicCheckpointer(
                self.checkpointer, __C.SOLVER.CHECKPOINT_PERIOD
            ),
            LazyEvalHook(__C.SOLVER.MAX_ITER // 2, __C.TEST.EVAL_PERIOD, _eval),
            d2.engine.hooks.PeriodicWriter(self.build_writers())
        ]
        # We need checkpointer and writer only for master process.
        return ret if dist.is_master_process() else [ret[0], ret[1], ret[3]]

    def run_step(self):
        r"""Extend ``run_step`` from ``SimpleTrainer``: support mixed precision."""

        # All this is similar to the super class method.
        assert self.model.training, "[DownstreamTrainer] is in eval mode!"
        data = next(self._data_loader_iter)

        # Pass one image at a time due to GPU constraints. Only do syncs in the
        # last iteration.
        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        # with amp.scale_loss(losses, self.optimizer) as scaled_losses:
        #     scaled_losses.backward()

        self.optimizer.step()


if __name__ == "__main__":

    _A = parser.parse_args()
    if not _A.serialization_dir:
        if _A.imagenet_backbone:
            raise ValueError("--serialization-dir required with --imagenet-backbone")

        CHECKPOINT_ITERATION = int(
            os.path.basename(_A.checkpoint_path).split("_")[-1][:-4]
        )
        _A.serialization_dir = os.path.join(
            os.path.dirname(_A.checkpoint_path), f"lvis_{CHECKPOINT_ITERATION}"
        )

    # Set up distributed environment - we use our `dist` utilities instead of
    # detectron2's utilities because it's easier with slurm.
    device_id = dist.init_distributed_env(_A.dist_backend) if _A.slurm else -1
    device = torch.device(f"cuda:{device_id}" if device_id != -1 else "cpu")
    if device_id != -1:
        d2.utils.comm._LOCAL_PROCESS_GROUP = torch.distributed.new_group(
            list(range(dist.get_world_size()))
        )

    # Create config with default values, then override from config file.
    # This is our config, not Detectron2 config.
    _C = Config(_A.config, _A.config_override)

    # We use `default_setup` from detectron2 to do some common setup, such as
    # logging, setting up serialization etc. For more info, look into source.
    _D2C = build_detectron2_config(_C, _A)
    default_setup(_D2C, _A)

    trainer = DownstreamTrainer(_D2C)

    # Load either imagenet weights or our pretrained weights.
    if _A.imagenet_backbone:
        trainer.checkpointer.load(_D2C.MODEL.WEIGHTS)
    else:
        # Initialize from a checkpoint, but only keep the visual module.
        model = PretrainingModelFactory.from_config(_C)
        model.load_state_dict(torch.load(_A.checkpoint_path))
        d2_weights = model.visual.detectron2_backbone_state_dict()
        trainer.checkpointer._load_model(d2_weights)
        del model

    trainer.train()
