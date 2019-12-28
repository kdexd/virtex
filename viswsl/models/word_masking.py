from typing import Any, Dict

import torch
from torch import nn

from viswsl.modules.fusion import Fusion


class WordMaskingModel(nn.Module):
    def __init__(self, visual, textual, fusion: Fusion):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.fusion = fusion

        self.output = nn.Linear(fusion.fused_feature_size, textual.vocab_size)
        self.loss = nn.CrossEntropyLoss(
            ignore_index=textual.padding_idx, reduction="none"
        )

    def forward(
        self,
        image: torch.Tensor,
        masked_tokens: torch.Tensor,
        masked_labels: torch.Tensor,
    ):
        batch_size = image.size(0)

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(image)

        # shape: (batch_size, ..., visual_feature_size)
        visual_features = visual_features.view(
            batch_size, self.visual.visual_feature_size, -1
        ).permute(0, 2, 1)

        # shape: (batch_size, num_caption_tokens, textual_feature_size)
        textual_features = self.textual(masked_tokens)

        # shape: (batch_size, num_caption_tokens, fused_feature_size)
        fused_features = self.fusion(visual_features, textual_features)

        # shape: (batch_size, num_caption_tokens, vocab_size)
        output_logits = self.output(fused_features)

        # Get predictions from logits, only the predictions at [MASK]ed
        # positions would be useful.
        predictions = torch.argmax(output_logits, dim=-1)

        output_dict: Dict[str, Any] = {
            "predictions": predictions,
            "loss": self.loss(
                output_logits.view(-1, output_logits.size(-1)),
                masked_labels.view(-1),
            ),
        }
        # Single scalar per batch for logging in training script.
        output_dict["loss_components"] = {
            "word_masking": output_dict["loss"].detach().mean()
        }
        return output_dict
