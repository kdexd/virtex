import csv
from typing import Any, Dict, Sequence

import sentencepiece as sp


class SentencePieceBPETokenizer(object):

    SP_SPACE = u"▁"

    def __init__(self, vocab_path: str, model_path: str):
        self.vocab_path = vocab_path
        self.model_path = model_path

        # Load pretrained tokenizer model.
        self.model = sp.SentencePieceProcessor()
        self.model.Load(model_path)

        # Load vocabulary mapping (and inverse mapping) between token and id.
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        with open(vocab_path, "r") as vocab_file:
            reader = csv.DictReader(
                vocab_file, delimiter="\t", fieldnames=["token", "logprob"]
            )
            for index, row in enumerate(reader):
                self._token_to_id[row["token"]] = index
                self._id_to_token[index] = row["token"]

    def __getstate__(self):
        r"""
        This magic method, along with ``__set_state__`` makes an object of this
        class picklable (and usable while data loading with multiple workers).
        """
        state_dict = self.__dict__.copy()
        state_dict["model"] = None
        return state_dict

    def __setstate__(self, state_dict: Dict[str, Any]):
        self.__dict__ = state_dict

        self.model = sp.SentencePieceProcessor()
        self.model.Load(self.model_path)

    def get_vocab_size(self) -> int:
        return len(self.model)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id["<unk>"])

    def id_to_token(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "<unk>")

    def encode(self, text: str) -> Sequence[int]:
        r"""Convert a text string to a list of integer token ids."""
        return self.model.EncodeAsIds(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        r"""Convert a sequence of token IDs to a text string."""
        return self.model.DecodeIds(token_ids)
