import torch
from torch import nn

from viswsl.modules.embedding import WordAndPositionalEmbedding


class LinguisticStream(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self._embedding = WordAndPositionalEmbedding(
            vocab_size, hidden_size, dropout_probability=0.1
        )

    def forward(self, tokens: torch.LongTensor):
        return self._embedding(tokens)
