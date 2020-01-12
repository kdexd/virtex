import copy
from typing import Any, Dict

import torch
from torch import nn

from viswsl.data import SentencePieceTokenizer, SentencePieceVocabulary
from viswsl.data.structures import CaptioningBatch
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
            self.backward_textual = copy.deepcopy(self.textual)
            self.backward_fusion = copy.deepcopy(self.fusion)

            # Tie the visual projection for both directions.
            self.fusion.projections._v_projection = (
                self.backward_fusion.projections._v_projection
            )
            # Tie word and position embeddings for both directions.
            self.backward_textual.embedding = self.textual.embedding

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
            "loss_components": {"captioning_forward": loss.detach()},
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
                captioning_backward=backward_loss.detach()
            )

        # During evaluation, get predictions from logits. Useful for logging.
        # Predictions from forward transformer will be shifted right by one
        # time-step, and vice-versa.
        if not self.training:
            predictions = torch.argmax(output_logits, dim=-1)
            output_dict["predictions"] = {"forward": predictions}

            if self.bidirectional:
                backward_predictions = backward_predictions = torch.argmax(
                    backward_output_logits, dim=-1
                )
                output_dict["predictions"]["backward"] = backward_predictions

        return output_dict

    def log_predictions(
        self,
        batch: CaptioningBatch,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        # fmt: off
        to_strtokens = lambda token_indices: [  # noqa: E731
            vocabulary.get_token_from_index(t.item())
            for t in token_indices if t.item() != vocabulary.pad_index
        ]
        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions["forward"]):
            predictions_str += f"""
                Caption tokens : {tokenizer.detokenize(to_strtokens(tokens))}
                Predictions (f): {tokenizer.detokenize(to_strtokens(preds))}

                """

        if self.bidirectional:
            for tokens, preds in zip(batch["noitpac_tokens"], predictions["backward"]):
                predictions_str += f"""
                    Noitpac tokens : {tokenizer.detokenize(to_strtokens(tokens))}
                    Predictions (b): {tokenizer.detokenize(to_strtokens(preds))}

                    """
        # fmt: on
        return predictions_str
