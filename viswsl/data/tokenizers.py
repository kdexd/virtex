from typing import List
import unicodedata

import sentencepiece as sp


class SentencePieceTokenizer(object):

    SP_SPACE = u"▁"

    def __init__(
        self,
        model_path: str,
        do_lower_case: bool = True,
        keep_accents: bool = False,
    ):
        self._model_path = model_path
        self._do_lower_case = do_lower_case
        self._keep_accents = keep_accents

        self._spm = sp.SentencePieceProcessor()
        self._spm.Load(model_path)

    def __len__(self):
        return len(self._spm)

    def tokenize(self, text: str) -> List[str]:
        r"""Convert a text string to a list of subword tokens (string)."""

        text = text.lower() if self._do_lower_case else text

        if not self._keep_accents:
            text = unicodedata.normalize("NFKD", text)
            text = "".join([c for c in text if not unicodedata.combining(c)])

        pieces = self._spm.EncodeAsPieces(text)
        return pieces

    def detokenize(self, tokens: List[str]) -> str:
        r"""Convert a list of subword tokens (string) to a text string."""
        out_string = "".join(tokens).replace(self.SP_SPACE, " ").strip()
        return out_string
