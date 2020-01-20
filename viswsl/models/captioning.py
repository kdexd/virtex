import copy
from typing import Any, Dict

import tokenizers as tkz
import torch
from torch import nn

from viswsl.data.structures import CaptioningBatch
from viswsl.modules.fusion import Fusion


class CaptioningModel(nn.Module):
    def __init__(
        self,
        visual,
        textual,
        fusion: Fusion,
        bidirectional: bool = False,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.fusion = fusion

        self.bidirectional = bidirectional
        self.tie_embeddings = tie_embeddings

        # Clone the textual and fusion modules for backward direction if
        # doing captioning in both directions (separately).
        if self.bidirectional:
            self.backward_textual = copy.deepcopy(self.textual)
            self.backward_fusion = copy.deepcopy(self.fusion)

        self.output: nn.Module = nn.Linear(
            self.fusion.fused_feature_size, self.textual.vocab_size
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=textual.padding_idx)

        self._tie_weights()

    def _tie_weights(self):
        r"""
        Tie weights at a few places to either save parameters, or simply where
        it makes more sense to have the same weights. For example, tie input
        and output word embeddings to save parameters. Have a same set of
        weights to project visual features (agnostic to textual components).
        This method is only called from :meth:`__init__`. Do not use it from
        outside the class definition.
        """

        # Tie input and output word embeddings to reduce parameters.
        # However, output embedding layer will learn its own bias.
        if (
            self.tie_embeddings
            and self.textual.textual_feature_size == self.fusion.fused_feature_size
        ):
            self.output.weight = self.textual.embedding.word_embedding.weight
        else:
            raise ValueError(
                "Expect input and output embeddings to be of same size for "
                f"tying weights, found {self.textual.textual_feature_size} and"
                f" {self.fusion.fused_feature_size} respectively."
            )

        if self.bidirectional:
            # Tie the visual projection for forward and backward directions.
            self.fusion.projections.visual = self.backward_fusion.projections.visual

            # Tie word and position embeddings for both directions.
            self.backward_textual.embedding = self.textual.embedding

    def forward(self, batch: CaptioningBatch):

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(batch["image"])

        # shape: (batch_size, ..., visual_feature_size)
        visual_features = visual_features.view(
            batch["image"].size(0), self.visual.visual_feature_size, -1
        ).permute(0, 2, 1)

        caption_tokens = batch["caption_tokens"]
        caption_lengths = batch["caption_lengths"]

        # shape: (batch_size, max_caption_length, textual_feature_size)
        textual_features = self.textual(caption_tokens, caption_lengths)

        # shape: (batch_size, num_caption_tokens, fused_feature_size)
        fused_features = self.fusion(visual_features, textual_features)

        # shape: (batch_size, num_caption_tokens, vocab_size)
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
        if self.bidirectional:
            backward_caption_tokens = batch["noitpac_tokens"]
            backward_textual_features = self.backward_textual(
                backward_caption_tokens, caption_lengths
            )
            backward_fused_features = self.backward_fusion(
                visual_features, backward_textual_features
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
            redundant_positions = caption_tokens[:, 1:] == self.textual.padding_idx
            predictions[redundant_positions] = self.textual.padding_idx
            output_dict["predictions"] = {"forward": predictions}

            if self.bidirectional:
                backward_predictions = backward_predictions = torch.argmax(
                    backward_output_logits, dim=-1
                )
                backward_predictions[redundant_positions] = self.textual.padding_idx
                output_dict["predictions"]["backward"] = backward_predictions

            output_dict["predictions"] = predictions

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
                Caption tokens : {tokenizer.decode(tokens)}
                Predictions (f): {tokenizer.decode(preds)}

                """

        if self.bidirectional:
            for tokens, preds in zip(
                batch["noitpac_tokens"], predictions["backward"]
            ):
                predictions_str += f"""
                Noitpac tokens : {tokenizer.decode(tokens)}
                Predictions (b): {tokenizer.decode(preds)}

                    """
        return predictions_str
