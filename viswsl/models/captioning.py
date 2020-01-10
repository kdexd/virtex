import copy
from typing import Any, Dict

import torch
from torch import nn

from viswsl.modules.fusion import Fusion


class CaptioningModel(nn.Module):
    def __init__(self, visual, textual, fusion: Fusion, bidirectional: bool = False):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.fusion = fusion

        # Clone the textual and fusion modules for backward direction if
        # doing captioning in both directions (separately).
        self.bidirectional = bidirectional

        if self.bidirectional:
            # Clone the textual branch and fusion branch.
            self.textual_backward = copy.deepcopy(self.textual)
            self.fusion_backward = copy.deepcopy(self.fusion)

            # Tie the visual projection for both directions.
            self.fusion.projections._v_projection = (
                self.fusion_backward.projections._v_projection
            )
            # Tie word and position embeddings for both directions.
            self.textual_backward.embedding = self.textual.embedding

        # Tie input and output word embeddings to reduce parameters.
        # Output embedding layer will also learn a bias.
        if textual.textual_feature_size == fusion.fused_feature_size:
            self.output: nn.Module = nn.Linear(
                fusion.fused_feature_size, textual.vocab_size
            )
            self.output.weight = self.textual.embedding.word_embedding.weight
        else:
            # Add an intermediate projection layer to `textual_feature_size`
            # if fused features have different size than textual features.
            self.output = nn.Sequential(
                nn.Linear(
                    fusion.fused_feature_size, textual.textual_feature_size, bias=False
                ),
                nn.Linear(
                    textual.textual_feature_size, textual.vocab_size
                )
            )
            self.output[0].weight.data.normal_(mean=0.0, std=0.02)
            self.output[-1].weight = self.textual.embedding.word_embedding.weight

        self.loss = nn.CrossEntropyLoss(ignore_index=textual.padding_idx)

    def forward(
        self,
        image: torch.Tensor,
        caption_tokens: torch.Tensor,
        caption_lengths: torch.Tensor,
    ):
        batch_size = image.size(0)
        max_caption_length = caption_lengths.max()

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(image)

        # shape: (batch_size, ..., visual_feature_size)
        visual_features = visual_features.view(
            batch_size, self.visual.visual_feature_size, -1
        ).permute(0, 2, 1)

        # Trim some token positions from the end if all captions are smaller
        # than max length.
        caption_tokens = caption_tokens[:, :max_caption_length]

        # shape: (batch_size, max_caption_length, textual_feature_size)
        textual_features = self.textual(caption_tokens, caption_lengths)

        # shape: (batch_size, num_caption_tokens, fused_feature_size)
        fused_features = self.fusion(visual_features, textual_features)

        # shape: (batch_size, num_caption_tokens, vocab_size)
        output_logits = self.output(fused_features)

        # Get predictions from logits, these will be shifted right by one
        # time-step (using forward transformer encoder).
        predictions = torch.argmax(output_logits, dim=-1)

        loss = self.loss(
            output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
            caption_tokens[:, 1:].contiguous().view(-1),
        )
        output_dict: Dict[str, Any] = {"predictions": predictions, "loss": loss}

        # Single scalar per batch for logging in training script.
        output_dict["loss_components"] = {"captioning_forward": loss.detach()}

        # Do captioning in backward direction.
        if self.bidirectional:
            backward_caption_tokens = self._reverse_flip(
                caption_tokens, caption_lengths
            )
            backward_textual_features = self.textual_backward(
                backward_caption_tokens, caption_lengths
            )
            backward_fused_features = self.fusion_backward(
                visual_features, backward_textual_features
            )
            backward_output_logits = self.output(backward_fused_features)
            backward_predictions = torch.argmax(backward_output_logits, dim=-1)

            backward_loss = self.loss(
                backward_output_logits[:, :-1]
                .contiguous()
                .view(-1, self.textual.vocab_size),
                backward_caption_tokens[:, 1:].contiguous().view(-1),
            )
            output_dict.update(backward_predictions=backward_predictions)
            output_dict["loss"] += backward_loss

            # Single scalar per batch for logging in training script.
            output_dict["loss_components"].update(
                captioning_backward=backward_loss.detach()
            )

        return output_dict

    @staticmethod
    def _reverse_flip(
        caption_tokens: torch.Tensor, caption_lengths: torch.Tensor
    ) -> torch.LongTensor:
        r"""
        Flips a (right-)padded batched tensor of caption tokens without
        affecting padded positions.

        Parameters
        ----------
        caption_tokens: torch.Tensor
            Batch of caption tokens, shape ``(batch_size, num_caption_tokens)``.
        caption_lengths: torch.Tensor
            Length of each caption in the batch, shape ``(batch_size, )``.

        Returns
        -------
        torch.Tensor
            Reversed captions, a tensor of same shape as ``caption_tokens``.
        """
        batch_size, max_caption_length = caption_tokens.size()
        flipped_caption_tokens = torch.flip(caption_tokens, [1])

        sequences = [
            flipped_caption_tokens[i, max_caption_length - length :]
            for i, length in enumerate(caption_lengths.tolist())
        ]
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
