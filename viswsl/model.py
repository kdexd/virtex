import torch
from torch import nn

# TODO (kd): have attention/fusion technique as a dependency injection.
from viswsl.modules.attention import ScaledDotProductAttention


class ViswslModel(nn.Module):
    # TODO (kd): Find a better name maybe?

    def __init__(self, visual, linguistic):
        super().__init__()
        self._visual = visual
        self._linguistic = linguistic

        # TODO (kd): Remove hardcoded values once this becomes a dependency
        # injection.
        self._attention = ScaledDotProductAttention(2048, linguistic.hidden_size)
        self._linear = nn.Linear(
            2048 + linguistic.hidden_size, linguistic.vocab_size
        )
        self._loss = nn.CrossEntropyLoss(ignore_index=linguistic.padding_idx)

    def forward(
        self,
        image: torch.Tensor,
        caption_tokens: torch.Tensor,
        masked_labels: torch.Tensor,
    ):
        # shape: (batch_size, 2048, 7, 7)
        image_features = self._visual(image)

        # shape: (batch_size, 49, 2048)
        image_features = image_features.view(-1, 2048, 49).permute(0, 2, 1)

        # shape: (batch_size, max_caption_length, hidden_size)
        output_hidden = self._linguistic(caption_tokens, masked_labels)

        # shape: (batch_size, max_caption_length, 2048)
        attended_features = self._attention(image_features, output_hidden)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self._linear(
            torch.cat((attended_features, output_hidden), dim=-1)
        )

        # Get predictions from logits, only the predictions at [MASK]ed
        # positions would be useful.
        predictions = torch.argmax(output_logits, dim=-1)
        output_dict = {"predictions": predictions}

        # Collapse dimensions: convert logits to (N, C), targets to (N,).
        output_dict["loss"] = self._loss(
            output_logits.view(-1, output_logits.size(-1)),
            masked_labels.view(-1),
        )
        return output_dict
