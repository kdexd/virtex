import argparse
import json
import os
from typing import Any, Dict, List

import dataflow as df
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(
    "Serialize a dataset of image-caption pairs to LMDB using dataflow."
)
parser.add_argument(
    "--images",
    default="data/coco/images/train2017",
    help="Path to a directory containing images of a dataset split.",
)
parser.add_argument(
    "--captions",
    default="data/coco/annotations/captions_train2017.json",
    help="Path to annotations file with captions for corresponding images.",
)
parser.add_argument(
    "--output",
    default="data/serialized/coco_train2017.lmdb",
    help="Path to store the file containing serialized dataset.",
)
parser.add_argument(
    "--num-procs",
    type=int,
    default=1,
    help="Number of processes for parallelization. Note that this may not "
    "preserve the order of instances as in annotations. Use 1 process to "
    "preserve order (might be slow).",
)
parser.add_argument(
    "--buffer-size",
    type=int,
    default=256,
    help="Number of datapoints to accumulate in a memory buffer. Should be "
    "ideally less than number of examples in dataset and take less memory "
    "than available RAM.",
)


class CocoCaptionsRawDataFlow(df.DataFlow):
    def __init__(
        self,
        images_dirpath: str,
        captions_filepath: str,
        dont_read_images: False,
    ):
        self._images = images_dirpath
        self._dont_read_images = dont_read_images

        _captions = json.load(open(captions_filepath))

        # Make a mapping between image_id and its filename.
        self._id_to_filename: Dict[int, str] = {
            im["id"]: im["file_name"] for im in _captions["images"]
        }

        # Make a mapping between image_id and its captions.
        self._id_to_captions: Dict[int, List[str]] = {}

        for ann in _captions["annotations"]:
            if ann["image_id"] not in self._id_to_captions:
                self._id_to_captions[ann["image_id"]] = []

            self._id_to_captions[ann["image_id"]].append(ann["caption"])

    def __iter__(self):

        for image_id in self._id_to_filename:
            image_path = os.path.join(
                self._images, self._id_to_filename[image_id]
            )
            captions = self._id_to_captions[image_id]

            if self._dont_read_images:
                yield image_path, captions
            else:
                yield self.read_image([image_path, captions])

    def __len__(self):
        return len(self._id_to_filename)

    @staticmethod
    def read_image(image_path_and_captions: List[Any]):

        # shape: (height, width, channels), dtype: uint8
        pil_image = Image.open(image_path_and_captions[0]).convert("RGB")
        image = np.asarray(pil_image)
        pil_image.close()

        # Return image (numpy array) and list of captions (as-is).
        return [image, image_path_and_captions[1]]


if __name__ == "__main__":

    _A = parser.parse_args()

    dflow = CocoCaptionsRawDataFlow(
        _A.images, _A.captions, dont_read_images=True
    )

    dflow = df.MultiProcessMapDataZMQ(
        dflow,
        num_proc=_A.num_procs,
        map_func=CocoCaptionsRawDataFlow.read_image,
        buffer_size=_A.buffer_size,
        strict=True,
    )

    # Resize shortest edge of image to 256 pixels.
    # Image is the first member in returned list by dataflow above.
    dflow = df.AugmentImageComponent(
        dflow, augmentors=[df.imgaug.ResizeShortestEdge(256)], index=0
    )

    os.makedirs(os.path.dirname(_A.output), exist_ok=True)
    df.LMDBSerializer.save(dflow, _A.output)

    # MultiprocessMapDataZMQ goes to a deadlock sometimes with strict mode.
    # Prompt to exit manually as a workaround.
    raise Exception("Serialization complete! Press Ctrl-C to exit.")
