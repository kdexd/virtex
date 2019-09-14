import math
import os
import random
from typing import List

import dataflow as df
import lmdb
import numpy as np
from torch.utils.data import get_worker_info, IterableDataset


class CocoCaptionsTrainDataset(IterableDataset):
    def __init__(self, lmdb_path: str, buffer_size: int = 8):

        self._lmdb_path = lmdb_path
        self._buffer_size = buffer_size

        # Get a list of "keys" in the LMDB file so we could shard the dataset
        # according to the number of worker processes.
        with lmdb.open(
            self._lmdb_path,
            subdir=os.path.isdir(self._lmdb_path),
            readonly=True,
            lock=False,
            readahead=True,
            map_size=1099511627776 * 2,
        ) as _lmdb_file:

            _txn = _lmdb_file.begin()
            self._keys: List[bytes] = df.utils.serialize.loads(
                _txn.get(b"__keys__")
            )

        # List of augmentations to be applied on each image after reading
        # from LMDB. This follows the standard augmentation steps of
        # (ImageNet pre-trained) ResNet models:
        #     1. Resize shortest edge to 256.
        #     2. Random crop a (224, 224) patch.
        #     3. Convert pixel intensities in [0, 1].
        #     4. Normalize image by mean pixel intensity and variance.
        #     5. Convert from HWC to CHW format.
        self._image_augmentor = df.imgaug.AugmentorList(
            [
                df.imgaug.ResizeShortestEdge(256),
                df.imgaug.RandomCrop(224),
                df.imgaug.ToFloat32(),
                df.imgaug.MapImage(
                    lambda image: (image - np.array([0.485, 0.456, 0.406]))
                    / np.array([0.229, 0.224, 0.225])
                ),
                df.imgaug.MapImage(
                    lambda image: np.transpose(image, (2, 0, 1))
                ),
            ]
        )

    def __len__(self):
        return len(self._keys)

    def __iter__(self):

        # ====================================================================
        # This code block will be executed just once, when an iterator of this
        # dataset is intialized: ``iter(dataset)`` or ``enumerate(dataset)``.
        # --------------------------------------------------------------------
        # Shard the dataset according to the number of workers.
        # Two members of interest:
        #     1. ``id: int = [0, n-1]``
        #     2. ``num_workers: int = n``.
        worker_info = get_worker_info()

        if worker_info is not None:
            _per_worker = int(
                math.ceil(len(self._keys) / worker_info.num_workers)
            )
            start = _per_worker * worker_info.id

            # Last worker may get less than ``_per_worker`` examples.
            end = min(len(self._keys), _per_worker * (worker_info.id + 1))
        else:
            # Single process data-loading, don't shard the dataset.
            start = 0
            end = len(self._keys)

        # Load examples from serialized LMDB (sharded by number of workers).
        # Read sequentially, random reads on large datasets may be expensive.
        dflow = df.LMDBData(
            self._lmdb_path, keys=self._keys[start:end], shuffle=False
        )

        # Decode bytes read from LMDB to Python objects.
        dflow = df.MapData(dflow, df.LMDBSerializer._deserialize_lmdb)

        # Keep a fixed-size buffer: examples will be pushed in this buffer and
        # randomly selected to make batches; a good proxy for random reads.
        dflow = df.LocallyShuffleData(dflow, self._buffer_size)
        dflow.reset_state()
        # ====================================================================

        for instance in dflow:
            image, captions = instance

            image = self._image_augmentor.augment(image)
            caption = random.choice(captions)

            yield image  # , caption
