import argparse
import json
import os
import tempfile
import unicodedata
from typing import List

import sentencepiece as sp


# fmt: off
parser = argparse.ArgumentParser(
    description="""Build a vocabulary out of captions corpus. This vocabulary
    would be a file which our tokenizer can understand.
    """
)
parser.add_argument(
    "-c", "--captions", default="datasets/coco/annotations/captions_train2017.json",
    help="Path to caption annotations file in COCO format.",
)
parser.add_argument(
    "-s", "--vocab-size", type=int, default=10000,
    help="Total desired size of our vocabulary.",
)
parser.add_argument(
    "-o", "--output-prefix", default="datasets/vocab/coco_10k",
    help="Prefix of the files to be saved. Two files will be saved: "
    "[prefix].model and [prefix].vocab",
)
parser.add_argument(
    "-l", "--do-lower-case", action="store_true",
    help="Whether to lower case the captions before forming vocabulary.",
)
parser.add_argument(
    "-a", "--keep-accents", action="store_true",
    help="Whether to keep accents before forming vocabulary (dropped by default).",
)
# fmt: on


def _read_captions(annotations_path: str) -> List[str]:
    r"""
    Given a path to annotation file, read it and return a list of captions.
    These are not processed by any means, returned from the file as-is.

    Args:
        annotations_path: Path to an annotations file containing captions.

    Returns:
        List of captions from this annotation file.
    """

    _annotations = json.load(open(annotations_path))

    captions: List[str] = []
    for ann in _annotations["annotations"]:
        captions.append(ann["caption"])

    return captions


if __name__ == "__main__":
    _A = parser.parse_args()
    captions: List[str] = _read_captions(_A.captions)

    # Lower case the captions and remove accents according to arguments.
    for i, caption in enumerate(captions):
        caption = caption.lower() if _A.do_lower_case else caption

        if not _A.keep_accents:
            caption = unicodedata.normalize("NFKD", caption)
            caption = "".join(
                [chr for chr in caption if not unicodedata.combining(chr)]
            )

        captions[i] = caption

    # Create a temporary directory and dump the captions corpus as a text file
    # with one caption per line. That's how sentencepiece wants its input.
    tmpdir_path = tempfile.mkdtemp()

    with open(os.path.join(tmpdir_path, "captions.txt"), "w") as captions_file:
        for caption in captions:
            captions_file.write(caption + "\n")

    # Padding/out-of-vocab token will be "<unk>" and ID 0 by default.
    # Add [SOS],[EOS] and [MASK] tokens. [MASK] will not be used during
    # captioning, but good to have to reuse vocabulary across pretext tasks.
    sp.SentencePieceTrainer.train(
        f" --input={os.path.join(tmpdir_path, 'captions.txt')}"
        f" --vocab_size={_A.vocab_size}"
        f" --model_prefix={_A.output_prefix}"
        " --model_type=bpe --character_coverage=1.0"
        " --bos_id=-1 --eos_id=-1"
        " --control_symbols=[SOS],[EOS],[MASK]"
    )
