import torch
from torch import nn

# TODO (kd): have attention/fusion technique as a dependency injection.
from viswsl.modules.attention import ScaledDotProductAttention


class WordMaskingModel(nn.Module):
    def __init__(self, visual, textual):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self._fused_projection_size = 2048 + textual.hidden_size

        self._attention = ScaledDotProductAttention(2048, textual.hidden_size)
        self._layer_norm = nn.LayerNorm(self._fused_projection_size, eps=1e-8)

        self._linear = nn.Linear(self._fused_projection_size, textual.vocab_size)
        self._loss = nn.CrossEntropyLoss(ignore_index=textual.padding_idx)

    def forward(
        self,
        image: torch.Tensor,
        masked_tokens: torch.Tensor,
        masked_labels: torch.Tensor,
    ):
        # shape: (batch_size, 2048, 7, 7)
        image_features = self.visual(image)

        # shape: (batch_size, 49, 2048)
        image_features = image_features.view(-1, 2048, 49).permute(0, 2, 1)

        # shape: (batch_size, max_caption_length, hidden_size)
        output_hidden = self.textual(masked_tokens)

        # shape: (batch_size, max_caption_length, 2048)
        attended_features = self._attention(image_features, output_hidden)

        # shape: (batch_size, max_caption_length, fused_projection_size)
        concatenated = torch.cat((attended_features, output_hidden), dim=-1)
        concatenated = self._layer_norm(concatenated)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self._linear(concatenated)

        # Get predictions from logits, only the predictions at [MASK]ed
        # positions would be useful.
        predictions = torch.argmax(output_logits, dim=-1)
        output_dict = {"predictions": predictions}

        # Collapse dimensions: convert logits to (N, C), targets to (N,).
        output_dict["loss"] = self._loss(
            output_logits.view(-1, output_logits.size(-1)), masked_labels.view(-1)
        )

        # Single scalar per batch for logging in training script.
        output_dict["loss_components"] = {
            "word_masking": output_dict["loss"].detach().mean()
        }
        return output_dict
