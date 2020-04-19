import argparse
import os
import pickle
import platform

import albumentations as alb
import lmdb
from tqdm import tqdm

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
    "-e", "--short-edge-size", type=int, default=None,
    help="""Resize shorter edge to this size (keeping aspect ratio constant)
    before serializing. Useful for saving disk memory, and faster read.
    If None, no images are resized."""
)
parser.add_argument(
    "-o", "--output", default="datasets/serialized/coco_train2017.lmdb",
    help="Path to store the file containing serialized dataset.",
)


if __name__ == "__main__":

    _A = parser.parse_args()
    os.makedirs(os.path.dirname(_A.output), exist_ok=True)

    dset = SimpleCocoCaptionsReader(_A.data_root, _A.split)

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
    for idx in tqdm(range(len(dset))):

        # Convert dict to an (image_id, image, captions) tuple for compactness.
        instance = dset[idx]

        # Resize image from instance and convert instance to tuple.
        image = instance["image"]
        if _A.short_edge_size is not None:
            image = resize(image=image)["image"]

        instance = (instance["image_id"], image, instance["captions"])

        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(instance, protocol=-1))
        txn.commit()

    db.sync()
    db.close()
