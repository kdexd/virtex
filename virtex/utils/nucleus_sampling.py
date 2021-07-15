r"""
Nucleus Sampling was introduced in the paper
`The Curious Case of Neural Text Degeneration <https://arxiv.org/abs/1904.09751>`_.
If you take it from here, make sure to cite them:

.. code-block:: text

    @inproceedings{,
        title={The Curious Case of Neural Text Degeneration},
        author={Ari Holtzman and Jan Buys and Li Du and Maxwell Forbes and Yejin Choi},
        journal={ICLR},
        year={2020}
    }

Some core parts of this code are adapted with minor modifications from Thomas Wolf's
gist: https://gist.githubusercontent.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
"""

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F


class AutoRegressiveNucleusSampling(object):
    r"""
    Implements the nucleus sampling for decoding captions. This class only works
    for auto-regressive models (Transformer-like), not recurrent models (LSTM-like).

    Parameters
    ----------
    eos_index: int
        The index of the end token (``[EOS]``) in vocabulary.
    max_steps: int, optional (default = 50)
        The maximum number of decoding steps.
    nucleus_size: float, optional (default = 0.9)
        Size of top-K nucleus for sampling.
    """

    def __init__(
        self,
        eos_index: int,
        max_steps: int = 50,
        nucleus_size: float = 0.9,
    ):
        super().__init__()
        self._eos_index = eos_index
        self.max_steps = max_steps
        self.nucleus_size = nucleus_size

    def search(
        self, start_predictions: torch.Tensor, step: Callable[..., torch.Tensor]
    ) -> Tuple[torch.Tensor, None]:

        batch_size = start_predictions.size()[0]

        # List of `(batch_size, )` tensors. One for each timestep.
        # This includes the start-of-sentence tokens, unlike the implementation
        # in `AutoregressiveBeamSearch`. We will remove them in the end.
        predictions: List[torch.Tensor] = [start_predictions]

        for timestep in range(self.max_steps):
            # Get the predictions from last timestep (most recent).
            # shape: (batch_size, )
            last_predictions = predictions[-1]

            # If every predicted token from the last step is end-of-sentence token,
            # then we can stop early.
            if (last_predictions == self._eos_index).all():
                break

            # Combine step predictions made so far into one tensor. This is our
            # "partial" caption input to the transformer.
            # shape: (batch_size, timestep + 1)
            predictions_so_far = torch.stack(predictions).permute(1, 0)

            # Take a step, get the distribution of logits from next timestep.
            # shape: (batch_size, num_classes)
            current_logits = step(predictions_so_far)

            # Sort logits in descending order to determine the nucleus.
            sorted_logits, sorted_idx = torch.sort(current_logits, descending=True)

            # Get cumulative softmax probabilites. For every instance in batch, a
            #  variable amount of tokens (N) will consitute the nucleus.
            # shape: (batch_size, num_classes)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Determine indices of tokens at the tail of distribution. These will be
            # removed from the nucleus.
            sorted_idx_to_remove = cumulative_probs > self.nucleus_size

            # Shift the indices to the right to keep the first token outside nucleus.
            sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
            sorted_idx_to_remove[..., 0] = 0

            # Set logits to large negative value to avoid sampling them. Iterate over
            # the batch of examples.
            for t in range(current_logits.size()[0]):
                idx_to_remove = sorted_idx[t][sorted_idx_to_remove[t]]
                current_logits[t][idx_to_remove] = -1e12

                # Set logits for last predicted token to a large negative value to
                # avoid repetition.
                current_logits[t][last_predictions[t]] = -1e12

            # Sample from the filtered distribution.
            # shape: (batch_size, num_classes)
            current_probs = F.softmax(current_logits, dim=-1)

            # shape: (batch_size, )
            current_predictions = torch.multinomial(current_probs, 1)
            current_predictions = current_predictions.view(batch_size)

            # Set current predicted tokens to be end-of-sentence for instances where
            # last prediction was also end-of-sentence token.
            current_predictions[last_predictions == self._eos_index] = self._eos_index

            predictions.append(current_predictions)

        # Remove start-of-sentence token from predictions, and collect them together.
        # shape: (batch_size, max_steps) .. or could be less than max_steps.
        all_predictions = torch.stack(predictions[1:]).permute(1, 0)

        # We don't return any logprobs of generated sequence with nucleus sampling,
        # unlike `AutoregressiveBeamSearch`.
        return all_predictions, None
