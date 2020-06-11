r"""
A *Reader* is a PyTorch :class:`~torch.utils.data.Dataset` which simply reads
data from disk and returns it almost as is. Readers defined here are used by
datasets in :mod:`virtex.data.datasets`.
"""
from collections import defaultdict
import glob
import json
import os
import pickle
import random
from typing import Dict, List, Tuple

import cv2
import lmdb
from loguru import logger
from torch.utils.data import Dataset


# Some simplified type renaming for better readability
ImageID = int
Captions = List[str]


class SimpleCocoCaptionsReader(Dataset):
    r"""
    A reader interface to read COCO Captions dataset and directly from official
    annotation files and return it unprocessed. We only use this for serializing
    the dataset to LMDB files, and use :class:`~virtex.data.readers.LmdbReader`
    in rest of the datasets.

    Parameters
    ----------
    root: str, optional (default = "datasets/coco")
        Path to the COCO dataset root directory.
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    """
    def __init__(self, root: str = "datasets/coco", split: str = "train"):

        image_dir = os.path.join(root, f"{split}2017")

        # Make a tuple of image id and its filename, get image_id from its
        # filename (assuming directory has images with names in COCO2017 format).
        image_filenames = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.id_filename: List[Tuple[ImageID, str]] = [
            (int(os.path.basename(name)[:-4]), name) for name in image_filenames
        ]

        # Make a mapping between image_id and its captions.
        _captions = json.load(
            open(os.path.join(root, "annotations", f"captions_{split}2017.json"))
        )
        self._id_to_captions: Dict[ImageID, Captions] = defaultdict(list)

        for ann in _captions["annotations"]:
            self._id_to_captions[ann["image_id"]].append(ann["caption"])

    def __len__(self):
        return len(self.id_filename)

    def __getitem__(self, idx: int):
        image_id, filename = self.id_filename[idx]

        # shape: (height, width, channels), dtype: uint8
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        captions = self._id_to_captions[image_id]

        return {"image_id": image_id, "image": image, "captions": captions}


class LmdbReader(Dataset):
    r"""
    A reader interface to read datapoints from a serialized LMDB file containing
    ``(image_id, image, caption)`` tuples. Optionally, one may specify a
    partial percentage of datapoints to use.

    .. note::

        When training in distributed setting, make sure each worker has SAME
        random seed because there is some randomness in selecting keys for
        training with partial dataset. If you wish to use a different seed for
        each worker, select keys manually outside of this class and use
        :meth:`set_keys`.

    .. note::

        Similar to :class:`~torch.utils.data.distributed.DistributedSampler`,
        this reader can shuffle the dataset deterministically at the start of
        epoch. Use :meth:`set_shuffle_seed` manually from outside to change the
        seed at every epoch.

    Parameters
    ----------
    lmdb_path: str
        Path to LMDB file with datapoints.
    shuffle: bool, optional (default = True)
        Whether to shuffle or not. If this is on, there will be one deterministic
        shuffle based on epoch before sharding the dataset (to workers).
    percentage: float, optional (default = 100.0)
        Percentage of datapoints to use. If less than 100.0, keys will be
        shuffled and first K% will be retained and use throughout training.
        Make sure to set this only for training, not validation.
    """

    def __init__(self, lmdb_path: str, shuffle: bool = True, percentage: float = 100):
        self.lmdb_path = lmdb_path
        self.shuffle = shuffle

        assert percentage > 0, "Cannot load dataset with 0 percent original size."
        self.percentage = percentage

        # fmt: off
        # Create an LMDB transaction right here. It will be aborted when this
        # class goes out of scope.
        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()

        # Form a list of LMDB keys numbered from 0 (as binary strings).
        self._keys = [
            f"{i}".encode("ascii") for i in range(env.stat()["entries"])
        ]
        # fmt: on

        # If data percentage < 100%, randomly retain K% keys. This will be
        # deterministic based on random seed.
        if percentage < 100.0:
            retain_k: int = int(len(self._keys) * percentage / 100.0)
            random.shuffle(self._keys)
            self._keys = self._keys[:retain_k]
            logger.info(f"Retained {retain_k} datapoints for training!")

        # A seed to deterministically shuffle at the start of epoch. This is
        # set externally through `set_shuffle_seed`.
        self.shuffle_seed = 0

    def set_shuffle_seed(self, seed: int):
        r"""Set random seed for shuffling data."""
        self.shuffle_seed = seed

    def get_keys(self) -> List[bytes]:
        r"""Return list of keys, useful while saving checkpoint."""
        return self._keys

    def set_keys(self, keys: List[bytes]):
        r"""Set list of keys, useful while loading from checkpoint."""
        self._keys = keys

    def __getstate__(self):
        r"""
        This magic method allows an object of this class to be pickable, useful
        for dataloading with multiple CPU workers. :attr:`db_txn` is not
        pickable, so we remove it from state, and re-instantiate it in
        :meth:`__setstate__`.
        """
        state = self.__dict__
        state["db_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx: int):
        datapoint_pickled = self.db_txn.get(self._keys[idx])
        image_id, image, captions = pickle.loads(datapoint_pickled)

        return image_id, image, captions
