from typing import Optional

import torch
from torch import nn


class WordAndPositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 512,
        max_sequence_length: int = 30,
        dropout_probability: float = 0.0,
        padding_idx: int = 0,
    ):
        super().__init__()

        self._word_embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx
        )
        # We provide no "padding index" for position embeddigs. We zero-out
        # the positional embeddings of padded positions as a post-processing,
        self._position_embedding = nn.Embedding(
            max_sequence_length, embedding_size
        )

        self._layer_norm = nn.LayerNorm(embedding_size, eps=1e-8)
        self._dropout = nn.Dropout(p=dropout_probability)
        self._padding_idx = padding_idx

    def forward(
        self,
        tokens: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
    ):
        batch_size, max_sequence_length = tokens.size()

        if positions is None:
            # Create position indices of the same size as token indices.
            positions = torch.arange(max_sequence_length, device=tokens.device)

            # shape: (batch_size, max_sequence_length)
            positions = positions.unsqueeze(0).expand_as(tokens)
        else:
            assert tokens.size() == positions.size(), "Expected tokens and "
            f"positions to have same size. Got {tokens.size()} and "
            f"{positions.size()} instead."

        # shape: (batch_size, max_sequence_length, embedding_size)
        word_embeddings = self._word_embedding(tokens)

        # shape: (batch_size, max_sequence_length, embedding_size)
        position_embeddings = self._position_embedding(positions)

        # Add the word and position embeddings, normalize them, and apply dropout.
        # shape: (batch_size, max_sequence_length, embedding_size)
        embeddings = self._layer_norm(word_embeddings + position_embeddings)
        embeddings = self._dropout(embeddings)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_sequence_length, 1)
        token_mask = (tokens != self._padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_sequence_length, embedding_size)
        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings
