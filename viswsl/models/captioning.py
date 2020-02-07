import copy
import functools
from typing import Any, Dict

import tokenizers as tkz
import torch
from torch import nn
from torch.nn import functional as F

from viswsl.data.structures import CaptioningBatch
from viswsl.modules.textual_stream import TextualStream
from viswsl.modules.visual_stream import VisualStream
from viswsl.utils.beam_search import AutoRegressiveBeamSearch


class CaptioningModel(nn.Module):
    def __init__(
        self,
        visual: VisualStream,
        textual: TextualStream,
        is_bidirectional: bool = False,
        beam_size: int = 5,
        max_decoding_steps: int = 30,
    ):
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
        self.is_bidirectional = is_bidirectional
        self.padding_idx = self.textual.padding_idx

        # Clone the textual module for backward direction if doing captioning
        # in both directions (separately).
        if self.is_bidirectional:
            self.backward_textual = copy.deepcopy(self.textual)
            # Tie word and position embeddings for both directions.
            self.backward_textual.embedding = self.textual.embedding

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        # Tie input and output word embeddings to reduce parameters.
        # However, output embedding layer will learn its own bias.
        self.output.weight = self.textual.embedding.words.weight

        # These boundary indices are needed for beam search.
        self.sos_index = textual.sos_index
        self.eos_index = textual.eos_index
        self.beam_search = AutoRegressiveBeamSearch(
            self.eos_index, beam_size=5, max_steps=max_decoding_steps
        )

    def forward(self, batch: CaptioningBatch):

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(batch["image"])
        batch_size = visual_features.size(0)

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
        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(textual_features)

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
            backward_output_logits = self.output(backward_textual_features)

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

        # During evaluation, get beam search predictions for forward model.
        # Predictions from forward transformer will be shifted right by one
        # time-step.
        if not self.training:
            start_predictions = projected_visual_features.new_full(
                (batch_size,), self.sos_index
            ).long()
            # Add image features as a default argument to match callable
            # signature accepted by beam search class (partial captions only).
            beam_search_step = functools.partial(
                self.beam_search_step, projected_visual_features
            )
            all_top_k_predictions, _ = self.beam_search.search(
                start_predictions, beam_search_step
            )
            best_beam = all_top_k_predictions[:, 0, :]
            output_dict["predictions"] = best_beam

        return output_dict

    def beam_search_step(
        self, projected_visual_features: torch.Tensor, partial_captions: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Given visual features and a batch of (assumed) partial captions, predict
        the distribution over vocabulary tokens for next time-step. This method
        is used by :class:`~viswsl.utils.beam_search.AutoRegressiveBeamSearch`.

        Parameters
        ----------
        projected_visual_features: torch.Tensor
            A tensor of shape ``(batch_size, ..., textual_feature_size)``
            with visual features already projected to ``textual_feature_size``.
        partial_captions: torch.Tensor
            A tensor of shape ``(batch_size * beam_size, timesteps)``
            containing tokens predicted so far -- one for each beam. We need all
            prior predictions because our model is auto-regressive.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size * beam_size, vocab_size)`` -- output
            distribution over tokens for next time-step.
        """

        batch_size, num_features, textual_feature_size = projected_visual_features.size()

        # Expand and repeat image features while doing beam search.
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            projected_visual_features = projected_visual_features.unsqueeze(1).repeat(
                1, beam_size, 1, 1
            )
            projected_visual_features = projected_visual_features.view(
                batch_size * beam_size, num_features, textual_feature_size
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a time-step. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, textual_feature_size)
        textual_features = self.textual(
            partial_captions, caption_lengths, projected_visual_features
        )
        # Keep features for last time-step only, we only care about those.
        # shape: (batch_size * beam_size, vocab_size)
        output_logits = self.output(textual_features[:, -1, :])

        # Return logprobs as required by `AutoRegressiveBeamSearch`.
        # shape: (batch_size * beam_size, vocab_size)
        next_logprobs = F.log_softmax(output_logits, dim=1)
        return next_logprobs

    def log_predictions(
        self, batch: CaptioningBatch, tokenizer: tkz.implementations.BaseTokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions):
            predictions_str += f"""
                Caption tokens : {tokenizer.decode(tokens.tolist())}
                Predictions (f): {tokenizer.decode(preds.tolist())}

                """
        return predictions_str
