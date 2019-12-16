import math
import os
from typing import Any, List, Tuple

import dataflow as df
import lmdb
import torch
from torch import distributed as dist
from torch.utils.data import get_worker_info


class ReadDatapointsFromLmdb(df.DataFlow):
    r"""
    A :class:`~dataflow.dataflow.DataFlow` to read datapoints from an LMDB
    file, from a subset of all datapoints. The LMDB file is logically split
    into multiple shards according to the number of parallel worker processes
    reading from it, in order to avoid duplicate datapoints in a batch.

    The dataset used (or any source of image-text pairs) does not matter. We
    serialize all of them using class:`~dataflow.serializers.LMDBSerializer`.

    Parameters
    ----------
    lmdb_path: str
    """

    def __init__(self, lmdb_path: str, shuffle: bool = False):
        self._lmdb_path = lmdb_path
        self._shuffle = shuffle

        # Get a list of "keys" in the LMDB file so we could shard the dataset.
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

        # For deterministically generating different shuffle ordes everytime
        # the read through starts again.
        self._read_epoch = 0

    def __len__(self):
        return len(self._keys)

    def __iter__(self):

        self._read_epoch += 1
        if self._shuffle:
            # Deterministically shuffle the dataset, differently everytime so.
            g = torch.Generator()
            g.manual_seed(self._read_epoch)
            indices = torch.randperm(len(self._keys), generator=g).tolist()
        else:
            indices = list(range(len(self._keys)))

        # ====================================================================
        # This code block will be executed just once, when an iterator of this
        # dataflow is intialized: ``iter(df)`` or ``enumerate(df)``.
        # --------------------------------------------------------------------
        # If we are doing distributed training, then first shard the LMDB
        # according to the number of GPU processes.
        world_size: int = dist.get_world_size() if dist.is_initialized() else 1
        world_rank: int = dist.get_rank() if dist.is_initialized() else 0

        # If not doing distributed training, this would be `len(self._keys)`.
        samples_per_gpu_process = int(math.ceil(len(self._keys) / world_size))

        # Further, shard the LMDB file according to the number of workers.
        # Two members of interest:
        #     1. ``id: int = [0, n-1]``
        #     2. ``num_workers: int = n``.
        worker_info = get_worker_info()

        if worker_info is not None:
            samples_per_worker = int(
                math.ceil(
                    len(indices) / (worker_info.num_workers * world_size)
                )
            )
            start = (
                world_rank * samples_per_gpu_process
                + worker_info.id * samples_per_worker
            )
            # Last worker may get less than ``_per_worker`` examples.
            end = min(len(indices), start + samples_per_worker)
        else:
            # Using single worker in dataloader, don't shard further.
            start = world_rank * samples_per_gpu_process
            end = min(len(indices), start + samples_per_gpu_process)

        keys_per_worker = [self._keys[i] for i in indices[start:end]]

        # Shuffle controlled outsie, set False here.
        pipeline = df.LMDBData(
            self._lmdb_path, keys=keys_per_worker, shuffle=False
        )
        # Decode bytes read from LMDB to Python objects.
        pipeline = df.MapData(pipeline, self._deserialize)
        pipeline.reset_state()
        # ====================================================================

        for image_id, instance in pipeline:
            image, captions = instance
            yield {"image_id": image_id, "image": image, "captions": captions}

    @staticmethod
    def _deserialize(datapoint: List[bytes]) -> Tuple[int, Any]:
        return (
            # LMDB key (from ``self._keys``) is simply an integer. Cast bytes
            # to int.
            int(datapoint[0]),
            df.utils.serialize.loads(datapoint[1]),
        )
