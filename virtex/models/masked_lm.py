from typing import Any, Dict

import torch
from torch import nn

from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.modules.textual_heads import TextualHead
from virtex.modules.visual_backbones import VisualBackbone


class MaskedLMModel(nn.Module):
    r"""
    A model to perform BERT-like masked language modeling. It is composed of a
    :class:`~virtex.modules.visual_backbones.VisualBackbone` and a
    :class:`~virtex.modules.textual_heads.TextualHead` on top of it.

    During training, the model received caption tokens with certain tokens
    replaced by ``[MASK]`` token, and it predicts these masked tokens based on
    surrounding context.

    Parameters
    ----------
    visual: virtex.modules.visual_backbones.VisualBackbone
        A :class:`~virtex.modules.visual_backbones.VisualBackbone` which
        computes visual features from an input image.
    textual: virtex.modules.textual_heads.TextualHead
        A :class:`~virtex.modules.textual_heads.TextualHead` which
        makes final predictions conditioned on visual features.
    """

    def __init__(self, visual: VisualBackbone, textual: TextualHead):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        r"""
        Given a batch of images and captions with certain masked tokens,
        predict the tokens at masked positions.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            A batch of images, ground truth caption tokens and masked labels.
            Possible set of keys: ``{"image_id", "image", "caption_tokens",
            "masked_labels", "caption_lengths"}``.

        Returns
        -------
        Dict[str, Any]

            A dict with the following structure, containing loss for optimization,
            loss components to log directly to tensorboard, and optionally
            predictions.

            .. code-block::

                {
                    "loss": torch.Tensor,
                    "loss_components": {"masked_lm": torch.Tensor},
                    "predictions": torch.Tensor
                }
        """

        # shape: (batch_size, channels, height, width)
        visual_features = self.visual(batch["image"])

        caption_tokens = batch["caption_tokens"]
        caption_lengths = batch["caption_lengths"]
        masked_labels = batch["masked_labels"]

        # shape: (batch_size, num_caption_tokens, vocab_size)
        output_logits = self.textual(visual_features, caption_tokens, caption_lengths)
        output_dict: Dict[str, Any] = {
            "loss": self.loss(
                output_logits.view(-1, output_logits.size(-1)), masked_labels.view(-1)
            )
        }
        # Single scalar per batch for logging in training script.
        output_dict["loss_components"] = {
            "masked_lm": output_dict["loss"].clone().detach()
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
        self, batch: Dict[str, torch.Tensor], tokenizer: SentencePieceBPETokenizer
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
