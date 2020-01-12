from typing import Any, Dict

import torch
from torch import nn

from viswsl.data import SentencePieceTokenizer, SentencePieceVocabulary
from viswsl.data.structures import WordMaskingBatch
from viswsl.modules.fusion import Fusion


class WordMaskingModel(nn.Module):
    def __init__(self, visual, textual, fusion: Fusion):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.fusion = fusion

        # Tie input and output word embeddings to reduce parameters.
        # Output embedding layer will also learn a bias.
        if textual.textual_feature_size == fusion.fused_feature_size:
            self.output = nn.Linear(fusion.fused_feature_size, textual.vocab_size)
            self.output.weight = self.textual.embedding.word_embedding.weight
        else:
            # Add an intermediate projection layer to `textual_feature_size`
            # if fused features have different size than textual features.
            self.output = nn.Sequential(
                nn.Linear(
                    fusion.fused_feature_size,
                    textual.textual_feature_size,
                    bias=False,
                ),
                nn.Linear(textual.textual_feature_size, textual.vocab_size),
            )
            self.output[0].weight.data.normal_(mean=0.0, std=0.02)
            self.output[-1].weight = self.textual.embedding.word_embedding.weight

        self.loss = nn.CrossEntropyLoss(ignore_index=textual.padding_idx)

    def forward(self, batch: WordMaskingBatch):
        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(batch["image"])

        # shape: (batch_size, ..., visual_feature_size)
        visual_features = visual_features.view(
            batch["image"].size(0), self.visual.visual_feature_size, -1
        ).permute(0, 2, 1)

        caption_tokens = batch["caption_tokens"]
        caption_lengths = batch["caption_lengths"]
        masked_labels = batch["masked_labels"]

        # shape: (batch_size, num_caption_tokens, textual_feature_size)
        textual_features = self.textual(caption_tokens, caption_lengths)

        # shape: (batch_size, num_caption_tokens, fused_feature_size)
        fused_features = self.fusion(visual_features, textual_features)

        # shape: (batch_size, num_caption_tokens, vocab_size)
        output_logits = self.output(fused_features)

        output_dict: Dict[str, Any] = {
            "loss": self.loss(
                output_logits.view(-1, output_logits.size(-1)),
                masked_labels.view(-1),
            )
        }
        # Single scalar per batch for logging in training script.
        output_dict["loss_components"] = {
            "word_masking": output_dict["loss"].detach()
        }
        # During evaluation, get predictions from logits. Useful for logging.
        # Only the predictions at [MASK]ed positions are relevant.
        if not self.training:
            predictions = torch.argmax(output_logits, dim=-1)
            redundant_positions = masked_labels == self.textual.padding_idx
            predictions[redundant_positions] = self.textual.padding_idx

            output_dict["predictions"] = predictions

        return output_dict

    def log_predictions(
        self,
        batch: WordMaskingBatch,
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
        for tokens, labels, preds in zip(
            batch["caption_tokens"], batch["masked_labels"], predictions,
        ):
            predictions_str += f"""
                Caption tokens : {tokenizer.detokenize(to_strtokens(tokens))}
                Masked Labels  : {tokenizer.detokenize(to_strtokens(labels))}
                Predictions    : {tokenizer.detokenize(to_strtokens(preds))}

                """
        # fmt: on
        return predictions_str
