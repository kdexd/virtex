import csv
import logging
from typing import Dict


logger = logging.getLogger(__name__)


class SentencePieceVocabulary(object):

    # NOTE (kd): You should be familiar to AllenNLP to know what's going on
    # here until they release v0.9 with PyTorch v1.2 support.

    # TODO (kd): Make this inherit AlleNNLP's Vocabulary class once the new
    # version releases, which starts supporting PyTorch v1.2.
    # For now, keep the API as close as AllenNLP's Vocabulary.

    def __init__(self, vocab_path: str):
        self._vocab_path = vocab_path

        # NOTE (kd): this vocabulary will have only one namespace named
        # "tokens", which will be non-padded.
        self._token_to_index: Dict[str, Dict[str, int]] = {"tokens": {}}
        self._index_to_token: Dict[str, Dict[int, str]] = {"tokens": {}}

        with open(vocab_path, "r") as vocab_file:
            reader = csv.DictReader(
                vocab_file, delimiter="\t", fieldnames=["token", "logprob"]
            )
            for index, row in enumerate(reader):
                self._token_to_index["tokens"][row["token"]] = index
                self._index_to_token["tokens"][index] = row["token"]

        # Short hand notations for convenience.
        self._oov_token = "<unk>"
        self._cls_token = "[CLS]"
        self._sep_token = "[SEP]"
        self._mask_token = "[MASK]"

    def get_index_to_token_vocabulary(self, namespace: str = "tokens"):
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = "tokens"):
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str = "tokens"):
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error("Namespace: %s", namespace)
                logger.error("Token: %s", token)
                raise

    def get_token_from_index(self, index: int, namespace: str = "tokens"):
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = "tokens"):
        return len(self._token_to_index[namespace])
