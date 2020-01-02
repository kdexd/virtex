import functools
from typing import Optional

import torch
from torch import nn


class WordAndPositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 512,
        max_sequence_length: int = 30,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx
        )
        # We provide no "padding index" for position embeddigs. We zero-out
        # the positional embeddings of padded positions as a post-processing,
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_size)
        self.layer_norm = nn.LayerNorm(
            embedding_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.LongTensor):
        batch_size, max_sequence_length = tokens.size()

        # shape: (batch_size, max_sequence_length, embedding_size)
        word_embeddings = self.word_embedding(tokens)

        positions = self.make_positions(
            batch_size, max_sequence_length, tokens.device
        )
        # shape: (batch_size, max_sequence_length, embedding_size)
        position_embeddings = self.position_embedding(positions)

        # Add the word and position embeddings, normalize them, and apply dropout.
        # shape: (batch_size, max_sequence_length, embedding_size)
        embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_sequence_length, 1)
        token_mask = (tokens != self.padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_sequence_length, embedding_size)
        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    @functools.lru_cache(maxsize=128)
    def make_positions(
        self, batch_size: int, max_sequence_length: int, device: torch.device
    ):
        r"""
        Make position indices for a tensor containing sequence. We wrap it in
        functools' ``lru_cache`` for a slight speedup.
        """
        # Create position indices of the same size as token indices.
        positions = torch.arange(max_sequence_length, device=device)

        # shape: (batch_size, max_sequence_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_sequence_length)
        return positions
