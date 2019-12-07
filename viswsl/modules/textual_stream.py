import torch
from torch import nn

from viswsl.modules.embedding import WordAndPositionalEmbedding


class DefaultTextualStream(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_attention_heads: int,
        num_layers: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers

        self._embedding = WordAndPositionalEmbedding(
            self.vocab_size, self.hidden_size, dropout_probability=0.1
        )

        _transformer_encoder_layer = nn.TransformerEncoderLayer(
            self.hidden_size, self.num_attention_heads, activation="gelu"
        )
        self._transformer_encoder = nn.TransformerEncoder(
            _transformer_encoder_layer, self.num_layers
        )
        self.padding_idx = padding_idx

    def forward(
        self, caption_tokens: torch.LongTensor, masked_labels: torch.Tensor
    ) -> torch.Tensor:

        # Form a mask, it is True for positions with padding token.
        # Transformer will ignore these positions for multi-headed attention.
        caption_mask = caption_tokens == self.padding_idx

        # shape: (batch_size, max_caption_length, embedding_size)
        token_embeddings = self._embedding(caption_tokens)

        # `TransformerEncoder` requires the sequence input as
        # (max_caption_length, batch_size, hidden_size). So we transpose the
        # first two dimensions of token embeddings, pass through encoder, and
        # later undo the transpose.
        token_embeddings = token_embeddings.transpose(0, 1)

        # shape: (max_caption_length, batch_size, hidden_size)
        output_hidden = self._transformer_encoder(
            token_embeddings, src_key_padding_mask=caption_mask
        )
        # shape: (batch_size, max_caption_length, hidden_size)
        output_hidden = output_hidden.transpose(0, 1)

        return output_hidden
