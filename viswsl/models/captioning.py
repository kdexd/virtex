import copy
from typing import Any, Dict

import tokenizers as tkz
import torch
from torch import nn

from viswsl.data.structures import CaptioningBatch
from viswsl.modules.fusion import Fusion
from viswsl.modules.textual_stream import TextualStream
from viswsl.modules.visual_stream import VisualStream


class CaptioningModel(nn.Module):
    def __init__(
        self,
        visual: VisualStream,
        textual: TextualStream,
        late_fusion: Fusion,
        is_bidirectional: bool = False,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.late_fusion = late_fusion

        # Linear layer to project image features to `textual_feature_size` to
        # facilitate decoder multi-head attention, fusion etc.
        self.visual_projection = nn.Linear(
            self.visual.visual_feature_size, self.textual.textual_feature_size
        )
        self.output = nn.Linear(
            self.late_fusion.fused_feature_size, self.textual.vocab_size
        )
        self.is_bidirectional = is_bidirectional
        self.padding_idx = self.textual.padding_idx

        # Clone the textual and late fusion modules for backward direction if
        # doing captioning in both directions (separately). o need to clone
        # early fusion, because direction doesn't play a role at that point.
        if self.is_bidirectional:
            self.backward_textual = copy.deepcopy(self.textual)
            self.backward_late_fusion = copy.deepcopy(self.late_fusion)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self._tie_weights()

    def _tie_weights(self):
        r"""
        Tie weights at a few places to either save parameters, or simply where
        it makes more sense to have the same weights. For example, tie input
        and output word embeddings to save parameters. This method is only
        called from :meth:`__init__`. Do not use it from outside.
        """

        # Tie input and output word embeddings to reduce parameters.
        # However, output embedding layer will learn its own bias.
        if self.textual.textual_feature_size == self.late_fusion.fused_feature_size:
            self.output.weight = self.textual.embedding.words.weight
        else:
            # Add an intermediate projection layer to `textual_feature_size`
            # if fused features have different size than textual features.
            self.output = nn.Sequential(
                nn.Linear(
                    self.late_fusion.fused_feature_size,
                    self.textual.textual_feature_size,
                    bias=False,
                ),
                nn.Linear(
                    self.textual.textual_feature_size, self.textual.vocab_size
                ),
            )
            self.output[0].weight.data.normal_(mean=0.0, std=0.02)
            self.output[-1].weight = self.textual.embedding.words.weight

        if self.is_bidirectional:
            # Tie word and position embeddings for both directions.
            self.backward_textual.embedding = self.textual.embedding

    def forward(self, batch: CaptioningBatch):

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(batch["image"])

        # shape: (batch_size, ..., visual_feature_size)
        visual_features = visual_features.view(
            batch["image"].size(0), self.visual.visual_feature_size, -1
        ).permute(0, 2, 1)

        # Now visual and textual features are of same size.
        # shape: (batch_size, ..., textual_feature_size)
        projected_visual_features = self.visual_projection(visual_features)

        caption_tokens = batch["caption_tokens"]
        caption_lengths = batch["caption_lengths"]

        # shape: (batch_size, max_caption_length, textual_feature_size)
        textual_features = self.textual(
            caption_tokens, caption_lengths, projected_visual_features
        )
        # shape: (batch_size, max_caption_length, fused_feature_size)
        fused_features = self.late_fusion(
            projected_visual_features, textual_features
        )
        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(fused_features)

        loss = self.loss(
            output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
            caption_tokens[:, 1:].contiguous().view(-1),
        )
        output_dict: Dict[str, Any] = {
            "loss": loss,
            # Single scalar per batch for logging in training script.
            "loss_components": {"captioning_forward": loss.clone().detach()},
        }
        # Do captioning in backward direction.
        if self.is_bidirectional:
            backward_caption_tokens = batch["noitpac_tokens"]

            backward_textual_features = self.backward_textual(
                backward_caption_tokens, caption_lengths, projected_visual_features
            )
            backward_fused_features = self.backward_late_fusion(
                projected_visual_features, backward_textual_features
            )
            backward_output_logits = self.output(backward_fused_features)

            backward_loss = self.loss(
                backward_output_logits[:, :-1]
                .contiguous()
                .view(-1, self.textual.vocab_size),
                backward_caption_tokens[:, 1:].contiguous().view(-1),
            )
            output_dict["loss"] += backward_loss

            # Single scalar per batch for logging in training script.
            output_dict["loss_components"].update(
                captioning_backward=backward_loss.clone().detach()
            )

        # During evaluation, get predictions from logits. Useful for logging.
        # Predictions from forward transformer will be shifted right by one
        # time-step, and vice-versa.
        if not self.training:
            predictions = torch.argmax(output_logits, dim=-1)[:, :-1]
            redundant_positions = caption_tokens[:, 1:] == self.padding_idx
            predictions[redundant_positions] = self.padding_idx
            output_dict["predictions"] = {"forward": predictions}

            if self.is_bidirectional:
                backward_predictions = backward_predictions = torch.argmax(
                    backward_output_logits, dim=-1
                )[:, :-1]
                backward_predictions[redundant_positions] = self.padding_idx
                output_dict["predictions"]["backward"] = backward_predictions

        return output_dict

    def log_predictions(
        self, batch: CaptioningBatch, tokenizer: tkz.implementations.BaseTokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions["forward"]):
            predictions_str += f"""
                Caption tokens : {tokenizer.decode(tokens.tolist())}
                Predictions (f): {tokenizer.decode(preds.tolist())}

                """

        if self.is_bidirectional:
            for tokens, preds in zip(
                batch["noitpac_tokens"], predictions["backward"]
            ):
                predictions_str += f"""
                Noitpac tokens : {tokenizer.decode(tokens.tolist())}
                Predictions (b): {tokenizer.decode(preds.tolist())}

                    """
        return predictions_str
