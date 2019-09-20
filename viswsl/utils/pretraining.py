import random
from typing import List, Tuple


def mask_some_tokens_randomly(
    tokens: List[str],
    masking_probability: float = 0.15,
    mask_token: str = "[MASK]",
    pad_token: str = "[UNK]",
    ignore_tokens: List[str] = [],
) -> Tuple[List[str], List[str]]:
    r"""
    Mask tokens for a single sentence. Tokens are expected to be strings.
    Extended Summary
    ----------------
    Tokens are masked as per the ``masking_probability``, which is 15% by
    default, following BERT. While masking, the token is replaced with
    ``mask_token`` 90% of the time, and retained the same 10% of the time.

    Parameters
    ----------
    tokens: List[str]
        A list of subword tokens of a sentence tokenized using
        :class:`~viswsl.data.tokenizers.SentencePieceTokenizer`.
    masking_probability: float, optional (default = 0.15)
        The proportion of tokens to be masked in the sentence.
    mask_token: str, optional (default = "[MASK]")
        The symbol of mask token as a string.
    pad_token: str, optional (default = "[UNK]")
        The symbol of padding token as a string. It is the same as out of
        vocabulary token (or the "unknown" token) in this codebase.
    ignore_tokens: List[str], optional (default = [])
        A list of tokens which shuld be ignored when considering to mask.
        These are usually special (control) tokens such as ``[CLS]``.
    """
    masked_labels = [pad_token] * len(tokens)

    for i, token in enumerate(tokens):

        if token != pad_token and token not in ignore_tokens:
            # Get a float in [0, 1) interval from a bernoulli distribution.
            # The probability of ``mask_flag < k`` is ``k``.
            mask_flag: float = random.random()
            if mask_flag <= masking_probability:

                # Replace with [MASK] 90% of the time.
                if random.random() >= 0.1:
                    masked_labels[i] = token
                    tokens[i] = mask_token

    return tokens, masked_labels
