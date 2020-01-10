import torch
from torch import nn

from viswsl.modules.embedding import WordAndPositionalEmbedding
from viswsl.modules.transformer import (
    PreNormTransformerEncoderLayer,
    BidirectionalTransformerEncoder,
    UnidirectionalTransformerEncoder,
)


class TransformerTextualStream(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        feedforward_size: int,
        attention_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        norm_type: str = "pre",
        activation: str = "gelu",
        is_bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.feedforward_size = feedforward_size
        self.attention_heads = attention_heads
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size, self.textual_feature_size, dropout=dropout
        )
        # Make encoder layer depending on whether it's a Pre-Norm or Post-Norm.
        EncoderLayerClass = (
            nn.TransformerEncoderLayer
            if norm_type == "post"
            else PreNormTransformerEncoderLayer
        )
        _encoder_layer = EncoderLayerClass(
            self.textual_feature_size,
            self.attention_heads,
            dim_feedforward=self.feedforward_size,
            dropout=dropout,
            activation=activation,
        )
        EncoderClass = (
            BidirectionalTransformerEncoder
            if is_bidirectional
            else UnidirectionalTransformerEncoder
        )
        self.encoder = EncoderClass(_encoder_layer, self.num_layers)
        self.apply(self.init_weights)

    @property
    def textual_feature_size(self):
        return self.hidden_size

    @staticmethod
    def init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self, caption_tokens: torch.Tensor, caption_lengths: torch.Tensor
    ) -> torch.Tensor:

        # Create a mask based on caption lengths, shape: (batch_size, )
        # Form a binary mask: it is True for padding positions.
        # These positions will be ignored for multi-headed attention.
        ones = torch.ones_like(caption_tokens)
        caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)

        # shape: (batch_size, max_caption_length, embedding_size)
        token_embeddings = self.embedding(caption_tokens)

        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = self.encoder(token_embeddings, token_mask=caption_mask)
        return textual_features
