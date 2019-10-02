from typing import Dict

import torch
from torch import nn

from viswsl.config import Config
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.modules.embedding import WordAndPositionalEmbedding


class LinguisticStream(nn.Module):
    # MOST OF THIS IS TEMPORARY. I AM ONLY TESTING OUT TRAINING LOOP.

    def __init__(
        self,
        vocabulary: SentencePieceVocabulary,
        hidden_size: int,
        num_attention_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self._vocabulary = vocabulary

        self.vocab_size = len(vocabulary)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers

        self._embedding = WordAndPositionalEmbedding(
            self.vocab_size, self.hidden_size, dropout_probability=0.1
        )

        _transformer_encoder_layer = nn.TransformerEncoderLayer(
            self.hidden_size, self.num_attention_heads
        )
        self._transformer_encoder = nn.TransformerEncoder(
            _transformer_encoder_layer, self.num_layers
        )
        self._linear = nn.Linear(self.hidden_size, self.vocab_size)

        self._loss = nn.CrossEntropyLoss(
            ignore_index=self._vocabulary.pad_index
        )

    def forward(
        self, caption_tokens: torch.LongTensor, masked_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # Form a mask, it is True for positions with padding token.
        # Transformer will ignore these positions for multi-headed attention.
        caption_mask = caption_tokens == self._vocabulary.pad_index

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

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self._linear(output_hidden)

        # Get the predictions from logits. Fill predictions to be the padding
        # token wherever there wasn't a [MASK].
        predictions = torch.argmax(output_logits, dim=-1)
        predictions = predictions.masked_fill(
            caption_tokens != self._vocabulary.mask_index,
            self._vocabulary.pad_index,
        )

        output_dict = {"predictions": predictions}
        if self.training:
            # Collapse dimensions: convert logits to (N, C), targets to (N,).
            output_dict["loss"] = self._loss(
                output_logits.view(-1, output_logits.size(-1)),
                masked_labels.view(-1),
            )

        return output_dict

    @classmethod
    def from_config(cls, config: Config):
        _C = config

        vocabulary = SentencePieceVocabulary(_C.DATA.VOCABULARY)
        return cls(
            vocabulary=vocabulary,
            hidden_size=_C.MODEL.LINGUISTIC.HIDDEN_SIZE,
            num_attention_heads=_C.MODEL.LINGUISTIC.NUM_ATTENTION_HEADS,
            num_layers=_C.MODEL.LINGUISTIC.NUM_LAYERS,
        )
