r"""
This Beam Search implementation is adapted with minor modifications from
`AllenNLP <https://github.com/allenai/allennlp/blob/master/allennlp/nn/beam_search.py>`_.

Thanks to the developers of AllenNLP!

**Update (v1.2):** The "backpointer" trick in Beam Search (as implemented in
AllenNLP) does not work well with autoregressive models (transformers). It is
now removed and it improves qualitative predictions and captioning metrics
(CIDEr/SPICE) for VirTex. Updated captioning results are on ArXiv v3. Refer
`CHANGELOG <https://github.com/kdexd/virtex/blob/master/CHANGELOG.md>`_ and
`Release Page <https://github.com/kdexd/virtex/releases/tag/v1.2>`_ for more
details.

Huge thanks to Nicolas Carion (@alcinos) and Aishwarya Kamath (@ashkamath) for
helping me fix this bug!
"""
from typing import Callable, Tuple
import warnings

import torch


class AutoRegressiveBeamSearch(object):
    r"""
    Implements the beam search algorithm for decoding the most likely captions.

    Parameters
    ----------
    end_index: int
        The index of the end token (``[EOS]``) in vocabulary.
    max_steps: int, optional (default = 50)
        The maximum number of decoding steps.
    beam_size: int, optional (default = 5)
        The width of the beam used.
    per_node_beam_size: int, optional (default = 2)
        The maximum number of candidates to consider per node, at each step in
        the search. Setting this parameter to a number smaller than `beam_size`
        may give better results, as it can introduce more diversity into the
        search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(
        self,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 5,
        per_node_beam_size: int = 2,
    ) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(
        self, start_predictions: torch.Tensor, step: Callable[..., torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Given a starting state and a step function, apply beam search to find
        the most likely target captions.

        .. note::

            If your step function returns ``-inf`` for some log probs
            (like if you're using a masked log-softmax) then some of the "best"
            captions returned may have ``-inf`` log probability. Specifically
            this happens when the beam size is smaller than the number of actions
            with finite log probability (non-zero probability) returned by the
            step function. Therefore if you're using a mask you may want to
            check the results from ``search`` and potentially discard captions
            with non-finite log probability.

        Parameters
        ----------
        start_predictions : torch.Tensor
            Tensor containing the initial predictions, shape ``(batch_size, )``.
            Usually the initial predictions are just the index of the start
            token (``[SOS]``) in the vocabulary.
        step : Callable[..., torch.Tensor]
            A function that is responsible for computing the next most likely
            tokens, given the past predictions. Predictions from all previous
            time-steps are required, not just the last time-step. The function
            should The function is expected to return a tensor of shape
            ``(group_size, target_vocab_size)`` containing the token log probs
            for the next step.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probs)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probs``
            has shape ``(batch_size, beam_size)``.
        """

        batch_size = start_predictions.size()[0]

        # List of `(batch_size, beam_size, length)` tensors.
        # Does not include the start symbols, which are implicit.
        predictions: torch.Tensor = torch.empty(
            (batch_size, self.beam_size, 0),
            dtype=torch.long, device=start_predictions.device
        )
        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probs = step(start_predictions)

        num_classes = start_class_log_probs.size()[1]

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probs, start_predicted_classes = start_class_log_probs.topk(
            self.beam_size
        )
        if (
            self.beam_size == 1
            and (start_predicted_classes == self._end_index).all()
        ):
            warnings.warn(
                "Empty captions predicted. You may want to increase beam "
                "size or ensure your step function is working properly.",
                RuntimeWarning,
            )
            return start_predicted_classes.unsqueeze(-1), start_top_log_probs

        # The log probs for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probs = start_top_log_probs

        # shape: (batch_size, beam_size, sequence_length)
        predictions = torch.cat([predictions, start_predicted_classes.unsqueeze(-1)], dim=-1)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[:, :, -1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            predictions_so_far = predictions.view(
                batch_size * self.beam_size, -1
            )          
            # shape: (batch_size * beam_size, num_classes)
            class_log_probs = step(predictions_so_far)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )
            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probs = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probs,
            )
            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probs, predicted_classes = cleaned_log_probs.topk(
                self.per_node_beam_size
            )
            # Here we expand the last log probs to `(batch_size * beam_size,
            # per_node_beam_size)` so that we can add them to the current log
            # probs for this timestep. This lets us maintain the log
            # probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probs = (
                last_log_probs.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.per_node_beam_size)
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probs = top_log_probs + expanded_last_log_probs

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probs.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # Append the predictions to the current beam.
            reshaped_beam = (
                predictions.view(batch_size * self.beam_size, 1, -1)
                .repeat(1, self.per_node_beam_size, 1)
                .reshape(batch_size, self.beam_size * self.per_node_beam_size, -1)
            )
            reshaped_beam = torch.cat([reshaped_beam, reshaped_predicted_classes.unsqueeze(-1)], dim=-1)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )
            predictions = reshaped_beam.gather(
                1, restricted_beam_indices.unsqueeze(-1).repeat(1,1,reshaped_beam.shape[-1])
            )

            # shape: (batch_size, beam_size)
            last_log_probs = restricted_beam_log_probs

        if not torch.isfinite(last_log_probs).all():
            warnings.warn(
                "Infinite log probs encountered. Some final captions may not "
                "make sense. This can happen when the beam size is larger than"
                " the number of valid (non-zero probability) transitions that "
                "the step function produces.",
                RuntimeWarning,
            )

        return predictions, last_log_probs