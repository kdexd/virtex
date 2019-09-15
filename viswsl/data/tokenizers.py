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


# class SentencePieceTokenizer(object):

#     def __init__(
#         do_lower_case=False,
#         remove_space=True,
#         keep_accents=False,
#         bos_token="<s>",
#         eos_token="</s>",
#         unk_token="<unk>",
#     ):
#         self.sp_model = spm.SentencePieceProcessor()
#         self.sp_model.Load(vocab_file)

#         self.bos_token = bos_token
#         self.bos_index = self.sp_model.PieceToId(self.bos_token)

#         self.eos_token = eos_token
#         self.eos_index = self.sp_model.PieceToId(self.eos_token)

#         self.unk_token = unk_token
#         self.unk_index = self.sp_model.PieceToId(self.unk_token)

#         self.vocab_file = vocab_file

#     def convert_tokens_to_ids(self, tokens):
#         """ Converts a single token or a sequence of tokens (str/unicode) in a integer id
#             (resp.) a sequence of ids, using the vocabulary.
#         """
#         if isinstance(tokens, str):
#             return self.sp_model.PieceToId(tokens)

#         ids = []
#         for token in tokens:
#             ids.append(self.sp_model.PieceToId(token))
#         if len(ids) > self.max_len:
#             logger.warning(
#                 "Token indices sequence length is longer than the specified maximum sequence length "
#                 "for this model ({} > {}). Running this sequence through the model will result in "
#                 "indexing errors".format(len(ids), self.max_len)
#             )
#         return ids

#     def convert_ids_to_tokens(self, ids):
#         """ Converts a single index or a sequence of indices (integers) in a token "
#             (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.
#         """
#         if isinstance(ids, int):
#             return self.sp_model.IdToPiece(ids)
#         tokens = []
#         for index in ids:
#             tokens.append(self.sp_model.IdToPiece(index))
#         return tokens
