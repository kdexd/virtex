import csv
from typing import Dict


class SentencePieceVocabulary(object):

    # TODO (kd): Make this inherit AlleNNLP's Vocabulary class once the new
    # version releases, which starts supporting PyTorch v1.2.
    # For now, keep the API as close as AllenNLP's Vocabulary.

    def __init__(self, vocab_path: str):
        self._vocab_path = vocab_path

        self._token_to_index: Dict[str, Dict[str, int]] = {}
        self._index_to_token: Dict[str, Dict[int, str]] = {}

        with open(vocab_path, "r") as vocab_file:
            reader = csv.DictReader(
                vocab_file, delimiter="\t", fieldnames=["token", "logprob"]
            )
            for index, row in enumerate(reader):
                self._token_to_index[row["token"]] = index
                self._index_to_token[index] = row["token"]

        # Short hand names for convenience. These will be accessed from
        # outside the class.
        self.pad_token = "<unk>"
        self.pad_index = self._token_to_index[self.pad_token]
        self.unk_token = "<unk>"
        self.unk_index = self._token_to_index[self.unk_token]
        self.cls_token = "[CLS]"
        self.cls_index = self._token_to_index[self.cls_token]
        self.sep_token = "[SEP]"
        self.sep_index = self._token_to_index[self.sep_token]
        self.mask_token = "[MASK]"
        self.mask_index = self._token_to_index[self.mask_token]

    @property
    def special_tokens(self):
        return [
            self.pad_token, self.unk_token, self.cls_token, self.sep_token,
            self.mask_token
        ]

    @property
    def special_indices(self):
        return [self.get_token_index(t) for t in self.special_tokens]

    def get_token_index(self, token: str):
        if token in self._token_to_index:
            return self._token_to_index[token]
        else:
            return self.unk_index

    def get_token_from_index(self, index: int):
        return self._index_to_token[index]

    def __len__(self):
        return len(self._token_to_index)
