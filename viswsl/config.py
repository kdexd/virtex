r"""
This module provides package-wide configuration management.
File adopted from several of my past projects:

- `https://github.com/kdexd/probnmn-clevr/blob/master/probnmn/config.py`_
- `https://github.com/nocaps-org/updown-baseline/blob/master/updown/config.py`_
"""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a
    nested dict-like structure, with nested keys accessible as attributes. It
    contains sensible default values for all the parameters, which may be
    overriden by (first) through a YAML file and (second) through a list of
    attributes and values.

    Extended Summary
    ----------------
    Modification of any parameter after instantiating an object of this class
    is not possible, so you must override required parameter values in either
    through ``config_yaml`` or ``config_override``.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override.
        This happens after overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        RANDOM_SEED: 42
        OPTIM:
          BATCH_SIZE: 512

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048])
    >>> _C.RANDOM_SEED  # default: 0
    42
    >>> _C.OPTIM.BATCH_SIZE  # default: 150
    2048

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

    DATA.MAX_CAPTION_LENGTH: 30
        Maximum length of captions as input to the linguistic stream. Captions
        longer than this will be truncated to maximum length.
    __________
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.RANDOM_SEED = 0

        self._C.DATA = CN()
        self._C.DATA.VOCABULARY = "data/coco_vocabulary.vocab"
        self._C.DATA.TOKENIZER = "data/coco_vocabulary.model"

        self._C.DATA.TRAIN_LMDB = "data/serialized/coco_train2017.lmdb"
        self._C.DATA.VAL_LMDB = "data/serialized/coco_val2017.lmdb"
        self._C.DATA.MAX_CAPTION_LENGTH = 30

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

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
        common_string: str = str(CN({"RANDOM_SEED": self._C.RANDOM_SEED})) + "\n"
        common_string += str(CN({"DATA": self._C.DATA})) + "\n"

        return common_string

    def __repr__(self):
        return self._C.__repr__()
