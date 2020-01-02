import functools

import torch
from torch import nn


class PreNormTransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""
    A variant of :class:`torch.nn.TransformerEncoderLayer` where layernorm is
    performed before self-attention and feedforward layers. This ``pre-norm``
    variant is used in GPT-2 and similar works, and is similar to pre-act
    variant of ResNet.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # fmt: off
        # We use the members (modules) from super-class, just the order of
        # operations is changed here. Layernorm first, then self-attention.
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)

        # Layernorm first, then transformation through feedforward network.
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        # fmt: on
        return src


class BidirectionalTansformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int):
        super().__init__(encoder_layer, num_layers)

    def forward(
        self, token_embeddings: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        # `TransformerEncoder` requires the sequence input as
        # (sequence_length, batch_size, hidden_size). So we transpose the
        # first two dimensions of tokens embeddings, pass through encoder, and
        # later undo the transpose.
        token_embeddings = token_embeddings.transpose(0, 1)

        # shape: (sequence_length, batch_size, hidden_size)
        outputs = super().forward(token_embeddings, src_key_padding_mask=token_mask)
        # shape: (batch_size, sequence_length, hidden_size)
        outputs = outputs.transpose(0, 1)
        return outputs


class UnidirectionalTransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        encoder_layer: nn.TransformerEncoderLayer,
        num_layers: int,
        backward: bool,
    ):
        super().__init__(encoder_layer, num_layers)
        self.backward = backward

    def forward(
        self, token_embeddings: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        # `TransformerEncoder` requires the sequence input as
        # (sequence_length, batch_size, hidden_size). So we transpose the
        # first two dimensions of tokens embeddings, pass through encoder, and
        # later undo the transpose.
        token_embeddings = token_embeddings.transpose(0, 1)
        sequence_length = token_embeddings.size(0)

        # Generate an additive mask based on direction of encoder: forward
        # encoder will mask the future, backward encoder will mask the past.
        # shape: (sequence_length, sequence_length)
        direction_mask = self._generate_mask(sequence_length)

        # shape: (sequence_length, batch_size, hidden_size)
        outputs = super().forward(
            token_embeddings,
            src_mask=direction_mask,
            src_key_padding_mask=token_mask,
        )

        # shape: (batch_size, sequence_length, hidden_size)
        outputs = outputs.transpose(0, 1)
        return outputs

    @functools.lru_cache(maxsize=10)
    def _generate_mask(self, size: int) -> torch.Tensor:
        r"""
        Generate a mask for "future" positions, useful when using this module
        for language modeling.

        Parameters
        ----------
        size: int
        """
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.tril(torch.ones(size, size), diagonal=-1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        if self.backward:
            mask = mask.flip(0, 1)

        return mask


ForwardTransformerEncoder = functools.partial(
    UnidirectionalTransformerEncoder, backward=False
)

BackwardTransformerEncoder = functools.partial(
    UnidirectionalTransformerEncoder, backward=True
)
