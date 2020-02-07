from typing import Any, Dict

import tokenizers as tkz
import torch
from torch import nn

from viswsl.data.structures import WordMaskingBatch
from viswsl.modules.textual_stream import TextualStream
from viswsl.modules.visual_stream import VisualStream


class WordMaskingModel(nn.Module):
    def __init__(self, visual: VisualStream, textual: TextualStream):
        super().__init__()
        self.visual = visual
        self.textual = textual

        # Linear layer to project image features to `textual_feature_size` to
        # facilitate decoder multi-head attention etc.
        self.visual_projection = nn.Linear(
            self.visual.visual_feature_size, self.textual.textual_feature_size
        )
        self.output = nn.Linear(
            self.textual.textual_feature_size, self.textual.vocab_size
        )
        self.padding_idx = self.textual.padding_idx
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        # Tie input and output word embeddings to reduce parameters.
        # However, output embedding layer will learn its own bias.
        self.output.weight = self.textual.embedding.words.weight

    def forward(self, batch: WordMaskingBatch):
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
        masked_labels = batch["masked_labels"]

        # shape: (batch_size, max_caption_length, textual_feature_size)
        textual_features = self.textual(
            caption_tokens, caption_lengths, projected_visual_features
        )
        # shape: (batch_size, num_caption_tokens, vocab_size)
        output_logits = self.output(textual_features)

        output_dict: Dict[str, Any] = {
            "loss": self.loss(
                output_logits.view(-1, output_logits.size(-1)),
                masked_labels.view(-1),
            )
        }
        # Single scalar per batch for logging in training script.
        output_dict["loss_components"] = {
            "word_masking": output_dict["loss"].clone().detach()
        }
        # During evaluation, get predictions from logits. Useful for logging.
        # Only the predictions at [MASK]ed positions are relevant.
        if not self.training:
            predictions = torch.argmax(output_logits, dim=-1)
            redundant_positions = masked_labels == self.padding_idx
            predictions[redundant_positions] = self.padding_idx

            output_dict["predictions"] = predictions

        return output_dict

    def log_predictions(
        self, batch: WordMaskingBatch, tokenizer: tkz.implementations.BaseTokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, labels, preds in zip(
            batch["caption_tokens"], batch["masked_labels"], predictions
        ):
            predictions_str += f"""
                Caption tokens : {tokenizer.decode(tokens.tolist())}
                Masked Labels  : {tokenizer.decode(labels.tolist())}
                Predictions    : {tokenizer.decode(preds.tolist())}

                """
        return predictions_str
