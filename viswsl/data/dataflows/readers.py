import math
import os
from typing import Any, Iterator, List, Tuple

import dataflow as df
import lmdb
from torch.utils.data import get_worker_info

from viswsl.types import LmdbDatapoint


class ReadDatapointsFromLmdb(df.DataFlow):
    r"""
    A :class:`~dataflow.dataflow.DataFlow` to read datapoints from an LMDB
    file, from a subset of all datapoints. The LMDB file is logically split
    into multiple shards according to the number of parallel worker processes
    reading from it, in order to avoid duplicate datapoints in a batch.

    Extended Summary
    ----------------
    The dataset used (or any source of image-text pairs) does not matter. We
    serialize all of them using class:`~dataflow.serializers.LMDBSerializer`.

    Parameters
    ----------
    lmdb_path: str
    buffer_size: int, optional (default = 8)
    """

    def __init__(self, lmdb_path: str, buffer_size: int = 8):
        self._lmdb_path = lmdb_path

        assert buffer_size > 0, "Buffer size cannot be zero or negative."
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

    def __len__(self):
        return len(self._keys)

    def __iter__(self) -> Iterator[LmdbDatapoint]:
        # ====================================================================
        # This code block will be executed just once, when an iterator of this
        # dataflow is intialized: ``iter(df)`` or ``enumerate(df)``.
        # --------------------------------------------------------------------
        # Shard the LMDB file according to the number of workers.
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
            # Single process data-loading, don't shard the file.
            start = 0
            end = len(self._keys)

        # Load examples from serialized LMDB (sharded by number of workers).
        # Read sequentially, random reads from large files may be expensive.
        pipeline = df.LMDBData(
            self._lmdb_path, keys=self._keys[start:end], shuffle=False
        )
        # Decode bytes read from LMDB to Python objects.
        pipeline = df.MapData(pipeline, self._deserialize)

        # Keep a fixed-size buffer: examples will be pushed in this buffer and
        # randomly selected to make batches; a good proxy for random reads.
        pipeline = df.LocallyShuffleData(pipeline, self._buffer_size)
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
