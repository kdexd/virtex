import argparse
import json
import unicodedata
from typing import List


# fmt: off
parser = argparse.ArgumentParser(
    description="""
    Build a caption corpus out of annotation files (COCO format). This corpus
    is a text file, one caption per line. it is used to train tokenizer
    (BPE, SentencePiece etc.).
    """
)
parser.add_argument(
    "-c", "--captions", default="datasets/coco/annotations/captions_train2017.json",
    help="Path to an annotation file containing captions (COCO format).",
)
parser.add_argument(
    "-o", "--output", default="datasets/coco_train2017_corpus.txt",
    help="Path to save the caption corpus.",
)

parser.add_argument_group("Configuration for pre-processing captions.")
parser.add_argument(
    "-l", "--do-lower-case", action="store_true",
    help="Whether to lower case the captions before adding to corpus.",
)
parser.add_argument(
    "-a", "--keep-accents", action="store_true",
    help="Whether to keep accent characters before adding to corpus.",
)
# fmt: on


def _read_captions(annotations_path: str) -> List[str]:
    r"""
    Given a path to annotation file, read it and return a list of captions.

    Parameters
    ----------
    annotations_path: str
        Path to an annotations file containing captions.

    Returns
    -------
    List[str]
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

    with open(_A.output, "w") as captions_file:
        for caption in captions:
            captions_file.write(caption + "\n")
