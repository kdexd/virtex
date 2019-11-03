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

    DATA.VOCABULARY: "data/coco_vocabulary.vocab"
        Path to a ``**.vocab`` file containing tokens. This file is used to
        instantiate :class:`~viswsl.data.vocabulary.SentencePieceVocabulary`.
    DATA.TOKENIZER: "data/coco_vocabulary.model"
        Path to a ``**.model`` file containing tokenizer model trained by
        `sentencepiece <https://www.github.com/google/sentencepiece>`_, used
        to instantiate :class:`~viswsl.data.tokenizer.SentencePieceTokenizer`.
    DATA.TRAIN_LMDB: data/serialized/coco_train2017.lmdb
        Path to an LMDB file containing training examples serialized as
        ``(image: np.ndarray, captions: List[str])``.
    DATA.VAL_LMDB: data/serialized/coco_val2017.lmdb
        Path to an LMDB file containing validation examples serialized as
        ``(image: np.ndarray, captions: List[str])``.
    DATA.NORMALIZE_IMAGE: True
        Whether to normalize the image by RGB color mean and variance.
    DATA.MAX_CAPTION_LENGTH: 30
        Maximum length of captions as input to the linguistic stream. Captions
        longer than this will be truncated to maximum length.
    __________

    MODEL:

    MODEL.VISUAL:
        Parameters defining the architecture of the visual stream.
    MODEL.VISUAL.NAME: "torchvision::resnext101_32x8d"
        Name of the visual stream model. Torchvision models supported for now.
    MODEL.VISUAL.PRETRAINED:
        Whether to initialize model from ImageNet pre-trained weights.
    _____

    MODEL.LINGUISTIC:
        Parameters defining the architecture of the linguistic stream.
    MODEL.LINGUISTIC.HIDDEN_SIZE: 512
        Size of the hidden state for the transformer.
    MODEL.LINGUISTIC.ATTENTION_HEADS: 8
        Number of attention heads for multi-headed attention.
    MODEL.LINGUISTIC.NUM_LAYERS: 6
        Number of layers in the transformer encoder.
    __________

    OPTIM:
        Optimization hyper-parameters, mostly relevant during training.

    OPTIM.BATCH_SIZE: 64
        Batch size during training and evaluation.
    OPTIM.BATCH_SIZE_MULTIPLIER: 1
        Number of batches to use for accumulating gradients before taking
        optimizer step. Useful to simulate large batch sizes.
    OPTIM.NUM_ITERATIONS: 400000
        Number of iterations to train for, batches are randomly sampled.
    OPTIM.LR: 1e-5
        Initial learning rate for optimizer. This linearly decays to zero till
        the end of training.
    OPTIM.WARMUP_STEPS: 2000
        Number of steps to perform LR warmup. Learning rate goes linearly from
        0 to ``OPTIM.LR`` for ``OPTIM.WARMUP_STEPS`` steps. A good rule of
        thumb is to set it as ``(2 / beta2 - 1)`` for Adam-like optimizers.
    OPTIM.WEIGHT_DECAY: 1e-3
        Weight decay co-efficient for optimizer.
    OPTIM.CLIP_GRADIENTS: 10
        Gradient clipping threshold to avoid exploding gradients.
    """

    def __init__(
        self, config_file: Optional[str] = None, override_list: List[Any] = []
    ):
        _C = CN()
        _C.RANDOM_SEED = 0

        _C.DATA = CN()
        _C.DATA.VOCABULARY = "data/coco_vocabulary.vocab"
        _C.DATA.TOKENIZER = "data/coco_vocabulary.model"
        _C.DATA.TRAIN_LMDB = "data/serialized/coco_train2017.lmdb"
        _C.DATA.VAL_LMDB = "data/serialized/coco_val2017.lmdb"
        _C.DATA.NORMALIZE_IMAGE = True
        _C.DATA.MAX_CAPTION_LENGTH = 30

        _C.PRETEXT = CN()
        _C.PRETEXT.MASKED_LM = CN()
        _C.PRETEXT.MASKED_LM.MASK_PROPORTION = 0.15
        _C.PRETEXT.MASKED_LM.MASK_PROBABILITY = 0.85
        _C.PRETEXT.MASKED_LM.REPLACE_PROBABILITY = 0.10

        _C.MODEL = CN()
        _C.MODEL.VISUAL = CN()
        _C.MODEL.VISUAL.NAME = "torchvision::resnext101_32x8d"
        _C.MODEL.VISUAL.PRETRAINED = False

        _C.MODEL.LINGUISTIC = CN()
        _C.MODEL.LINGUISTIC.HIDDEN_SIZE = 512
        _C.MODEL.LINGUISTIC.NUM_ATTENTION_HEADS = 8
        _C.MODEL.LINGUISTIC.NUM_LAYERS = 6

        _C.OPTIM = CN()
        _C.OPTIM.OPTIMIZER_NAME = "adamw"
        _C.OPTIM.BATCH_SIZE = 64
        _C.OPTIM.BATCH_SIZE_MULTIPLIER = 1
        _C.OPTIM.NUM_ITERATIONS = 400000
        _C.OPTIM.LR = 1e-3
        _C.OPTIM.WARMUP_STEPS = 2000
        _C.OPTIM.WEIGHT_DECAY = 1e-3
        _C.OPTIM.CLIP_GRADIENTS = 10

        # Override parameter values from YAML file first, then from override
        # list.
        self._C = _C
        if config_file is not None:
            self._C.merge_from_file(config_file)
        self._C.merge_from_list(override_list)

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

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        common_string: str = str(
            CN({"RANDOM_SEED": self._C.RANDOM_SEED})
        ) + "\n"
        common_string += str(CN({"DATA": self._C.DATA})) + "\n"
        common_string += str(CN({"PRETEXT": self._C.DATA})) + "\n"
        common_string += str(CN({"MODEL": self._C.MODEL})) + "\n"
        common_string += str(CN({"OPTIM": self._C.OPTIM})) + "\n"

        return common_string

    def __repr__(self):
        return self._C.__repr__()
