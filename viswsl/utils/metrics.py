from collections import defaultdict
import json
import os
from subprocess import Popen, PIPE, check_call
import tempfile
from typing import Any, Dict, List, Union

import numpy as np


# Some type annotations for better readability
ImageID = int
Caption = str


# Punctuations to be removed from the sentences (PTB style)).
# fmt: off
PUNCTS = [
    "''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!",
    ",", ":", "-", "--", "...", ";",
]
# fmt: on

# Some constants for CIDEr and SPICE metrics.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPICE_JAR = f"{CURRENT_DIR}/assets/SPICE-1.0/spice-1.0.jar"
CACHE_DIR = f"{CURRENT_DIR}/assets/cache"


class CocoCaptionsEvaluator(object):
    def __init__(self, gt_annotations: Union[str, List[Any]]):

        # Read annotations from the path (if path is provided).
        if isinstance(gt_annotations, str):
            gt_annotations = json.load(open(gt_annotations))["annotations"]

        # Keep a mapping from image id to a list of captions.
        self.ground_truth: Dict[ImageID, List[Caption]] = defaultdict(list)
        for ann in gt_annotations:
            self.ground_truth[ann["image_id"]].append(ann["caption"])  # type: ignore

        self.ground_truth = tokenize(self.ground_truth)

    def evaluate(self, preds):

        if isinstance(preds, str):
            preds = json.load(open(preds))

        res = {ann["image_id"]: [ann["caption"]] for ann in preds}
        res = tokenize(res)

        # Remove IDs from predictions which are not in GT.
        common_image_ids = self.ground_truth.keys() & res.keys()
        res = {k: v for k, v in res.items() if k in common_image_ids}

        # Add dummy entries for IDs absent in preds, but present in GT.
        for k in self.ground_truth:
            res[k] = res.get(k, [""])

        cider_score = cider(self.ground_truth, res)
        spice_score = spice(self.ground_truth, res)

        return {"CIDEr": 100 * cider_score, "SPICE": 100 * spice_score}


def tokenize(
    image_id_to_captions: Dict[ImageID, List[Caption]]
) -> Dict[ImageID, List[Caption]]:
    r"""
    Given a mapping of image id to a list of corrsponding captions, tokenize
    captions in place according to Penn Treebank Tokenizer. This method assumes
    the presence of Stanford CoreNLP JAR file in directory of this module.
    """
    # Path to the Stanford CoreNLP JAR file.
    CORENLP_JAR = "stanford-corenlp-3.4.1.jar"

    # Prepare data for Tokenizer: write captions to a text file, one per line.
    image_ids = [k for k, v in image_id_to_captions.items() for _ in range(len(v))]
    sentences = "\n".join(
        [c.replace("\n", " ") for k, v in image_id_to_captions.items() for c in v]
    )
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(sentences.encode())
    tmp_file.close()

    # fmt: off
    # Tokenize sentences. We use the JAR file for tokenization.
    command = [
        "java", "-cp", CORENLP_JAR, "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines", "-lowerCase", tmp_file.name
    ]
    tokenized_captions = (
        Popen(command, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=PIPE)
        .communicate(input=sentences.rstrip())[0]
        .decode()
        .split("\n")
    )
    # fmt: on
    os.remove(tmp_file.name)

    # Map tokenized captions back to their image IDs.
    image_id_to_tokenized_captions: Dict[ImageID, List[Caption]] = defaultdict(list)
    for image_id, caption in zip(image_ids, tokenized_captions):
        image_id_to_tokenized_captions[image_id].append(
            " ".join([w for w in caption.rstrip().split(" ") if w not in PUNCTS])
        )

    return image_id_to_tokenized_captions


def cider(
    ground_truth: Dict[ImageID, List[Caption]],
    predictions: Dict[ImageID, List[Caption]],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    r"""Compute CIDEr score given ground truth captions and predictions."""

    # -------------------------------------------------------------------------
    def to_ngrams(sentence: str, n: int = 4):
        r"""Convert a sentence into n-grams and their counts."""
        words = sentence.split()
        counts = defaultdict(int)  # type: ignore
        for k in range(1, n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i : i + k])
                counts[ngram] += 1
        return counts

    def counts2vec(cnts, document_frequency, log_reference_length):
        r"""Function maps counts of ngram to vector of tfidf weights."""
        vec = [defaultdict(float) for _ in range(n)]
        length = 0
        norm = [0.0 for _ in range(n)]
        for (ngram, term_freq) in cnts.items():
            df = np.log(max(1.0, document_frequency[ngram]))
            # tf (term_freq) * idf (precomputed idf) for n-grams
            vec[len(ngram) - 1][ngram] = float(term_freq) * (
                log_reference_length - df
            )
            # Compute norm for the vector: will be used for computing similarity
            norm[len(ngram) - 1] += pow(vec[len(ngram) - 1][ngram], 2)

            if len(ngram) == 2:
                length += term_freq
        norm = [np.sqrt(nn) for nn in norm]
        return vec, norm, length

    def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
        r"""Compute the cosine similarity of two vectors."""
        delta = float(length_hyp - length_ref)
        val = np.array([0.0 for _ in range(n)])
        for nn in range(n):
            for (ngram, count) in vec_hyp[nn].items():
                val[nn] += (
                    min(vec_hyp[nn][ngram], vec_ref[nn][ngram]) * vec_ref[nn][ngram]
                )

            val[nn] /= (norm_hyp[nn] * norm_ref[nn]) or 1
            val[nn] *= np.e ** (-(delta ** 2) / (2 * sigma ** 2))
        return val

    # -------------------------------------------------------------------------

    ctest = [to_ngrams(predictions[image_id][0]) for image_id in ground_truth]
    crefs = [
        [to_ngrams(gt) for gt in ground_truth[image_id]] for image_id in ground_truth
    ]
    # Build document frequency and compute IDF.
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1

    # Compute log reference length.
    log_reference_length = np.log(float(len(crefs)))

    scores = []
    for test, refs in zip(ctest, crefs):
        # Compute vector for test captions.
        vec, norm, length = counts2vec(
            test, document_frequency, log_reference_length
        )
        # Compute vector for ref captions.
        score = np.array([0.0 for _ in range(n)])
        for ref in refs:
            vec_ref, norm_ref, length_ref = counts2vec(
                ref, document_frequency, log_reference_length
            )
            score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)

        score_avg = np.mean(score)
        score_avg /= len(refs)
        score_avg *= 10.0
        scores.append(score_avg)

    return np.mean(scores)


def spice(
    ground_truth: Dict[ImageID, List[Caption]],
    predictions: Dict[ImageID, List[Caption]],
) -> float:
    r"""Compute SPICE score given ground truth captions and predictions."""

    # Prepare temporary input file for the SPICE scorer.
    input_data = [
        {
            "image_id": image_id,
            "test": predictions[image_id][0],
            "refs": ground_truth[image_id],
        }
        for image_id in ground_truth
    ]
    # Create a temporary directory and dump input file to SPICE.
    temp_dir = tempfile.mkdtemp()
    INPUT_PATH = os.path.join(temp_dir, "input_file.json")
    OUTPUT_PATH = os.path.join(temp_dir, "output_file.json")
    json.dump(input_data, open(INPUT_PATH, "w"))

    # fmt: off
    # Run the command to execute SPICE jar.
    os.makedirs(CACHE_DIR, exist_ok=True)
    spice_cmd = [
        "java", "-jar", "-Xmx8G", SPICE_JAR, INPUT_PATH,
        "-cache", CACHE_DIR, "-out", OUTPUT_PATH, "-subset", "-silent",
    ]
    check_call(spice_cmd, cwd=CURRENT_DIR)
    # fmt: on

    # Read and process results
    results = json.load(open(OUTPUT_PATH))
    image_id_to_scores = {item["image_id"]: item["scores"] for item in results}
    spice_scores = [
        np.array(item["scores"]["All"]["f"]).astype(float) for item in results
    ]
    return np.mean(spice_scores)
