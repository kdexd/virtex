r"""
Parts of this class are adopted from several of my past projects:

- `https://github.com/kdexd/probnmn-clevr/blob/master/probnmn/config.py`_
- `https://github.com/nocaps-org/updown-baseline/blob/master/updown/config.py`_
"""
from typing import Any, List, Optional

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    This class provides package-wide configuration management. It is a
    nested dict-like structure with nested keys accessible as attributes. It
    contains sensible default values, which can be modified by (first) a YAML
    file and (second) a list of attributes and values.

    Note
    ----
    The instantiated object is "immutable" - any modification is prohibited.
    You must override required parameter values either through ``config_file``
    or ``override_list``.

    Parameters
    ----------
    config_file: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override.
        This happens after overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        RANDOM_SEED: 42
        OPTIM:
          WEIGHT_DECAY: 1e-2

    >>> _C = Config("config.yaml", ["OPTIM.WEIGHT_DECAY", 1e-4])
    >>> _C.RANDOM_SEED  # default: 0
    42
    >>> _C.OPTIM.WEIGHT_DECAY  # default: 1e-3
    1e-4

    Attributes
    ----------
    RANDOM_SEED: 0
        Random seed for NumPy and PyTorch, important for reproducibility.
    __________

    DATA:
        Collection of required data paths for training and evaluation. All
        these are assumed to be relative to project root directory. If
        elsewhere, symlinking is recommended.

    DATA.MAX_CAPTION_LENGTH: 30
        Maximum length of captions as input to the textual stream. Captions
        longer than this will be truncated to maximum length.
    __________

    MODEL:

    MODEL.VISUAL:
        Parameters defining the architecture of the visual stream.
    MODEL.VISUAL.NAME: "torchvision::resnet50"
        Name of the visual stream model. Torchvision models supported for now.
    MODEL.VISUAL.PRETRAINED:
        Whether to initialize model from ImageNet pre-trained weights.
    _____

    MODEL.TEXTUAL:
        Parameters defining the architecture of the textual stream.
    MODEL.TEXTUAL.NUM_LAYERS: 6
        Number of layers in the transformer encoder.
    __________

    OPTIM:
        Optimization hyper-parameters, mostly relevant during training.

    OPTIM.OPTIMIZER_NAME: adamw
        One of ``["sgd", "adam", "adamw"]``.
    OPTIM.NUM_ITERATIONS: 1000000
        Number of iterations to train for, batches are randomly sampled.
    OPTIM.BATCH_SIZE: 256
        Total batch size for training/evaluation (to be divided across GPUs).

    OPTIM.LR: 1e-3
        Initial learning rate for optimizer. This linearly decays to zero till
        the end of training.
    OPTIM.WARMUP_STEPS: 10000
        Number of steps to perform LR warmup. Learning rate goes linearly from
        0 to ``OPTIM.LR`` for ``OPTIM.WARMUP_STEPS`` steps. A good rule of
        thumb is to set it as ``(2 / 1 - beta2)`` for Adam-like optimizers, or
        5-10% of total number of iterations.
    OPTIM.WEIGHT_DECAY: 1e-4
        Weight decay co-efficient for optimizer.
    OPTIM.CLIP_GRAD_NORM: 10
        Threshold to clamp gradients for avoiding exploding gradients.
    """

    def __init__(
        self, config_file: Optional[str] = None, override_list: List[Any] = []
    ):
        _C = CN()
        _C.RANDOM_SEED = 0
        _C.FP16_OPT = 0

        _C.DATA = CN()
        _C.DATA.ROOT = "datasets/coco"
        _C.DATA.TOKENIZER_VOCAB = "datasets/vocab/coco_10k.vocab"
        _C.DATA.TOKENIZER_MODEL = "datasets/vocab/coco_10k.model"

        _C.DATA.IMAGE_CROP_SIZE = 224
        _C.DATA.MAX_CAPTION_LENGTH = 30

        _C.DATA.USE_SINGLE_CAPTION = False
        _C.DATA.USE_PERCENTAGE = 100.0

        _C.DATA.IMAGE_TRANSFORM_TRAIN = [
            "random_resized_crop",
            "horizontal_flip",
            "color_jitter",
            "normalize",
        ]
        _C.DATA.IMAGE_TRANSFORM_VAL = [
            "smallest_resize",
            "center_crop",
            "normalize",
        ]

        _C.PRETEXT = CN()
        _C.PRETEXT.WORD_MASKING = CN()
        _C.PRETEXT.WORD_MASKING.MASK_PROPORTION = 0.15
        _C.PRETEXT.WORD_MASKING.MASK_PROBABILITY = 0.85
        _C.PRETEXT.WORD_MASKING.REPLACE_PROBABILITY = 0.10

        _C.MODEL = CN()
        _C.MODEL.NAME = "bicaptioning"
        _C.MODEL.DROPOUT = 0.1

        _C.MODEL.VISUAL = CN()
        _C.MODEL.VISUAL.NAME = "torchvision::resnet50"
        _C.MODEL.VISUAL.FEATURE_SIZE = 2048
        _C.MODEL.VISUAL.PRETRAINED = False
        _C.MODEL.VISUAL.FROZEN = False

        _C.MODEL.TEXTUAL = CN()
        _C.MODEL.TEXTUAL.NAME = "transformer_postnorm"
        _C.MODEL.TEXTUAL.HIDDEN_SIZE = 512
        _C.MODEL.TEXTUAL.ATTENTION_HEADS = 8
        _C.MODEL.TEXTUAL.FEEDFORWARD_SIZE = 2048
        _C.MODEL.TEXTUAL.NUM_LAYERS = 1

        _C.OPTIM = CN()
        _C.OPTIM.NUM_ITERATIONS = 500000
        _C.OPTIM.OPTIMIZER_NAME = "sgd"
        _C.OPTIM.CLIP_GRAD_NORM = 10

        _C.OPTIM.SGD_MOMENTUM = 0.9
        _C.OPTIM.USE_LOOKAHEAD = False
        _C.OPTIM.LOOKAHEAD_STEPS = 5
        _C.OPTIM.LOOKAHEAD_ALPHA = 0.5

        _C.OPTIM.BATCH_SIZE = 256
        _C.OPTIM.WEIGHT_DECAY = 0.0001
        _C.OPTIM.NO_DECAY = ".*textual.*(norm.*|bias)"

        _C.OPTIM.LR = 0.001
        _C.OPTIM.CNN_LR = 0.2
        _C.OPTIM.WARMUP_STEPS = 10000
        _C.OPTIM.LR_DECAY_NAME = "cosine"
        _C.OPTIM.LR_GAMMA = 0.1
        _C.OPTIM.LR_STEPS = []

        # Override parameter values from YAML file first, then from override
        # list, then add derived params.
        self._C = _C
        if config_file is not None:
            self._C.merge_from_file(config_file)
        self._C.merge_from_list(override_list)

        self.add_derived_params()

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def add_derived_params(self):
        r"""Add parameters with values derived from existing parameters."""

        # ---------------------------------------------------------------------
        # Set textual stream architecture if specified in string.
        # For example: "transformer_postnorm::L6_H768_A12_F3072":
        #     L = layers, H = hidden_size, A = attention_heads, F= feedforward_size
        tstream_name_parts = self._C.MODEL.TEXTUAL.NAME.split("::")[-1].split("_")
        for name_part in tstream_name_parts:
            if name_part[0] == "L":
                self._C.MODEL.TEXTUAL.NUM_LAYERS = int(name_part[1:])
            elif name_part[0] == "H":
                self._C.MODEL.TEXTUAL.HIDDEN_SIZE = int(name_part[1:])
            elif name_part[0] == "A":
                self._C.MODEL.TEXTUAL.ATTENTION_HEADS = int(name_part[1:])
            elif name_part[0] == "F":
                self._C.MODEL.TEXTUAL.FEEDFORWARD_SIZE = int(name_part[1:])
        # ---------------------------------------------------------------------

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        return self._C.__str__()

    def __repr__(self):
        return self._C.__repr__()
