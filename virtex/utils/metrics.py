r"""
This module is a collection of metrics commonly used during pretraining and
downstream evaluation. Two main classes here are:

- :class:`TopkAccuracy` used for ImageNet linear classification evaluation.
- :class:`CocoCaptionsEvaluator` used for caption evaluation (CIDEr and SPICE).

Parts of this module (:meth:`tokenize`, :meth:`cider` and :meth:`spice`) are
adapted from `coco-captions evaluation code <https://github.com/tylin/coco-caption>`_.
"""
from collections import defaultdict
import json
import os
from subprocess import Popen, PIPE, check_call
import tempfile
from typing import Any, Dict, List

import numpy as np
import torch


class TopkAccuracy(object):
    r"""
    An accumulator for Top-K classification accuracy. This accumulates per-batch
    accuracy during training/validation, which can retrieved at the end. Assumes
    integer labels and predictions.

    .. note::

        If used in :class:`~torch.nn.parallel.DistributedDataParallel`, results
        need to be aggregated across GPU processes outside this class.

    Parameters
    ----------
    top_k: int, optional (default = 1)
        ``k`` for computing Top-K accuracy.
    """

    def __init__(self, top_k: int = 1):
        self._top_k = top_k
        self.reset()

    def reset(self):
        r"""Reset counters; to be used at the start of new epoch/validation."""
        self.num_total = 0.0
        self.num_correct = 0.0

    def __call__(self, predictions: torch.Tensor, ground_truth: torch.Tensor):
        r"""
        Update accumulated accuracy using the current batch.

        Parameters
        ----------
        ground_truth: torch.Tensor
            A tensor of shape ``(batch_size, )``, an integer label per example.
        predictions : torch.Tensor
            Predicted logits or log-probabilities of shape
            ``(batch_size, num_classes)``.
        """

        if self._top_k == 1:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
        else:
            top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

        correct = top_k.eq(ground_truth.unsqueeze(-1)).float()

        self.num_total += ground_truth.numel()
        self.num_correct += correct.sum()

    def get_metric(self, reset: bool = False):
        r"""Get accumulated accuracy so far (and optionally reset counters)."""
        if self.num_total > 1e-12:
            accuracy = float(self.num_correct) / float(self.num_total)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy


class CocoCaptionsEvaluator(object):
    r"""A helper class to evaluate caption predictions in COCO format. This uses
    :meth:`cider` and :meth:`spice` which exactly follow original COCO Captions
    evaluation protocol.

    Parameters
    ----------
    gt_annotations_path: str
        Path to ground truth annotations in COCO format (typically this would
        be COCO Captions ``val2017`` split).
    """

    def __init__(self, gt_annotations_path: str):
        gt_annotations = json.load(open(gt_annotations_path))["annotations"]

        # Keep a mapping from image id to a list of captions.
        self.ground_truth: Dict[int, List[str]] = defaultdict(list)
        for ann in gt_annotations:
            self.ground_truth[ann["image_id"]].append(ann["caption"])

        self.ground_truth = tokenize(self.ground_truth)

    def evaluate(self, preds: List[Dict[str, Any]]) -> Dict[str, float]:
        r"""Compute CIDEr and SPICE scores for predictions.

        Parameters
        ----------
        preds: List[Dict[str, Any]]
            List of per instance predictions in COCO Captions format:
            ``[ {"image_id": int, "caption": str} ...]``.

        Returns
        -------
        Dict[str, float]
            Computed metrics; a dict with keys ``{"CIDEr", "SPICE"}``.
        """
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

        cider_score = cider(res, self.ground_truth)
        spice_score = spice(res, self.ground_truth)

        return {"CIDEr": 100 * cider_score, "SPICE": 100 * spice_score}


def tokenize(image_id_to_captions: Dict[int, List[str]]) -> Dict[int, List[str]]:
    r"""
    Given a mapping of image id to a list of corrsponding captions, tokenize
    captions in place according to Penn Treebank Tokenizer. This method assumes
    the presence of Stanford CoreNLP JAR file in directory of this module.
    """
    # Path to the Stanford CoreNLP JAR file.
    CORENLP_JAR = (
        "assets/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar"
    )

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
    # Punctuations to be removed from the sentences (PTB style)).
    # fmt: off
    PUNCTS = [
        "''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?",
        "!", ",", ":", "-", "--", "...", ";",
    ]
    # fmt: on
    image_id_to_tokenized_captions: Dict[int, List[str]] = defaultdict(list)
    for image_id, caption in zip(image_ids, tokenized_captions):
        image_id_to_tokenized_captions[image_id].append(
            " ".join([w for w in caption.rstrip().split(" ") if w not in PUNCTS])
        )

    return image_id_to_tokenized_captions


def cider(
    predictions: Dict[int, List[str]],
    ground_truth: Dict[int, List[str]],
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
    predictions: Dict[int, List[str]], ground_truth: Dict[int, List[str]]
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
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SPICE_JAR = f"{CURRENT_DIR}/assets/SPICE-1.0/spice-1.0.jar"
    CACHE_DIR = f"{CURRENT_DIR}/assets/cache"
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
