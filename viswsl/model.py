import torch
from torch import nn


class ViswslModel(nn.Module):
    # TODO (kd): Find a better name maybe?

    def __init__(self, visual, linguistic):
        super().__init__()
        self._visual = visual
        self._linguistic = linguistic

        self._visual_projection = nn.Linear(2048, linguistic.hidden_size)
        self._linear = nn.Linear(linguistic.hidden_size, linguistic.vocab_size)

        self._loss = nn.CrossEntropyLoss(ignore_index=linguistic.padding_idx)

    def forward(
        self,
        image: torch.Tensor,
        caption_tokens: torch.Tensor,
        masked_labels: torch.Tensor,
    ):

        # shape: (batch_size, 2048)
        image_features = self._visual(image)

        # shape: (batch_size, hidden_size)
        image_features = self._visual_projection(image_features)

        # shape: (batch_size, max_caption_length, hidden_size)
        output_hidden = self._linguistic(caption_tokens, masked_labels)

        image_features = image_features.unsqueeze(1).repeat(
            1, output_hidden.size(1), 1
        )

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self._linear(output_hidden * image_features)

        # Get predictions from logits, only the predictions at [MASK]ed
        # positions would be useful.
        predictions = torch.argmax(output_logits, dim=-1)
        output_dict = {"predictions": predictions}

        if self.training:
            # Collapse dimensions: convert logits to (N, C), targets to (N,).
            output_dict["loss"] = self._loss(
                output_logits.view(-1, output_logits.size(-1)),
                masked_labels.view(-1),
            )

        return output_dict
