import csv
from typing import Any, Dict, List

import sentencepiece as sp


class SentencePieceBPETokenizer(object):
    r"""
    A tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`_
    with BPE sub-routine. It encodes caption strings into list of tokens.

    Parameters
    ----------
    model_path: str
        Path to the ``.model`` file trained by SentencePiece.
    """
    SP_SPACE = u"▁"

    def __init__(self, model_path: str):
        self.model_path = model_path

        # Load pretrained tokenizer model.
        self.model = sp.SentencePieceProcessor()
        self.model.Load(model_path)

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
        # Since tokenizer uses subword regularization, one token may break down to multiple IDs.
        # Keep trying till we get a single ID.
        return self.model.piece_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        r"""Get string token of an integer ID (``<unk>`` if does not exist)."""
        return self.model.id_to_piece(token_id)

    def encode(self, text: str) -> List[int]:
        r"""Convert a text string to a list of integer token ids."""
        return self.model.EncodeAsIds(text)

    def decode(self, token_ids: List[int]) -> str:
        r"""Convert a sequence of token IDs to a text string."""
        return self.model.DecodeIds(token_ids)
