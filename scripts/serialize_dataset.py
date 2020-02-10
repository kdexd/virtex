import argparse
import json
import os
from typing import Dict, List

import dataflow as df
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(
    "Serialize a dataset of image-caption pairs to LMDB using dataflow."
)
parser.add_argument(
    "--images",
    default="datasets/coco/images/train2017",
    help="Path to a directory containing images of a dataset split.",
)
parser.add_argument(
    "--captions",
    default="datasets/coco/annotations/captions_train2017.json",
    help="Path to annotations file with captions for corresponding images.",
)
parser.add_argument(
    "--output",
    default="datasets/serialized/coco_train2017.lmdb",
    help="Path to store the file containing serialized dataset.",
)


class CocoCaptionsRawDataFlow(df.DataFlow):
    def __init__(
        self,
        images_dirpath: str,
        captions_filepath: str,
        dont_read_images: bool = False,
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

            try:
                image = self.read_image(image_path)
                yield image, captions
            except Exception as e:
                print(f"Failed image {image_id} with {e.__class__.__name__}")
                continue

    def __len__(self):
        return len(self._id_to_filename)

    @staticmethod
    def read_image(image_path: str):
        # shape: (height, width, channels), dtype: uint8
        pil_image = Image.open(image_path).convert("RGB")
        image = np.asarray(pil_image)
        pil_image.close()
        return image


if __name__ == "__main__":

    _A = parser.parse_args()

    dflow = CocoCaptionsRawDataFlow(_A.images, _A.captions)

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
