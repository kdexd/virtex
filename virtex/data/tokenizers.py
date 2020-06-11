import csv
from typing import Any, Dict, List

import sentencepiece as sp


class SentencePieceBPETokenizer(object):
    r"""
    A tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`_
    with BPE sub-routine. It encodes caption strings into list of tokens.

    Parameters
    ----------
    vocab_path: str
        Path to the ``.vocab`` file trained by SentencePiece.
    model_path: str
        Path to the ``.model`` file trained by SentencePiece.
    """
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
        This magic method, along with ``__setstate__`` makes an object of this
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
        r"""Return number of tokens in vocabulary (including special tokens)."""
        return len(self.model)

    def token_to_id(self, token: str) -> int:
        r"""Get integer ID of a string token (``<unk>`` if does not exist)."""
        return self._token_to_id.get(token, self._token_to_id["<unk>"])

    def id_to_token(self, token_id: int) -> str:
        r"""Get string token of an integer ID (``<unk>`` if does not exist)."""
        return self._id_to_token.get(token_id, "<unk>")

    def encode(self, text: str) -> List[int]:
        r"""Convert a text string to a list of integer token ids."""
        return self.model.EncodeAsIds(text)

    def decode(self, token_ids: List[int]) -> str:
        r"""Convert a sequence of token IDs to a text string."""
        return self.model.DecodeIds(token_ids)
