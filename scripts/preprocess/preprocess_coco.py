import argparse
import os
import pickle
import platform
from typing import Any, List

import albumentations as alb
import lmdb
from tqdm import tqdm
from torch.utils.data import DataLoader

from virtex.data.readers import SimpleCocoCaptionsReader


# fmt: off
parser = argparse.ArgumentParser("Serialize a COCO Captions split to LMDB.")
parser.add_argument(
    "-d", "--data-root", default="datasets/coco",
    help="Path to the root directory of COCO dataset.",
)
parser.add_argument(
    "-s", "--split", choices=["train", "val"],
    help="Which split to process, either `train` or `val`.",
)
parser.add_argument(
    "-b", "--batch-size", type=int, default=16,
    help="Batch size to process and serialize data. Set as per CPU memory.",
)
parser.add_argument(
    "-j", "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for data loading.",
)
parser.add_argument(
    "-e", "--short-edge-size", type=int, default=None,
    help="""Resize shorter edge to this size (keeping aspect ratio constant)
    before serializing. Useful for saving disk memory, and faster read.
    If None, no images are resized."""
)
parser.add_argument(
    "-o", "--output", default="datasets/serialized/coco_train2017.lmdb",
    help="Path to store the file containing serialized dataset.",
)


def collate_fn(instances: List[Any]):
    r"""Collate function for data loader to return list of instances as-is."""
    return instances


if __name__ == "__main__":

    _A = parser.parse_args()
    os.makedirs(os.path.dirname(_A.output), exist_ok=True)

    dloader = DataLoader(
        SimpleCocoCaptionsReader(_A.data_root, _A.split),
        batch_size=_A.batch_size,
        num_workers=_A.cpu_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    # Open an LMDB database.
    # Set a sufficiently large map size for LMDB (based on platform).
    map_size = 1099511627776 * 2 if platform.system() == "Linux" else 1280000
    db = lmdb.open(
        _A.output, map_size=map_size, subdir=False, meminit=False, map_async=True
    )

    # Transform to resize shortest edge and keep aspect ratio same.
    if _A.short_edge_size is not None:
        resize = alb.SmallestMaxSize(max_size=_A.short_edge_size, always_apply=True)

    # Serialize each instance (as a dictionary). Use `pickle.dumps`. Key will
    # be an integer (cast as string) starting from `0`.
    INSTANCE_COUNTER: int = 0

    for idx, batch in enumerate(tqdm(dloader)):

        txn = db.begin(write=True)

        for instance in batch:
            image = instance["image"]
            width, height, channels = image.shape

            # Resize image from instance and convert instance to tuple.
            if _A.short_edge_size is not None and min(width, height) > _A.short_edge_size:
                image = resize(image=image)["image"]

            instance = (instance["image_id"], instance["image"], instance["captions"])
            txn.put(
                f"{INSTANCE_COUNTER}".encode("ascii"),
                pickle.dumps(instance, protocol=-1)
            )
            INSTANCE_COUNTER += 1

        txn.commit()

    db.sync()
    db.close()
