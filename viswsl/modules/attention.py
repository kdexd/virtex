from typing import Optional

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        value_size: int,
        query_size: int,
        key_size: Optional[int] = None,
        provide_keys_externally: bool = False,
    ):
        super().__init__()
        self.query_size = query_size
        self.value_size = value_size
        self.key_size = key_size or query_size
        self._provide_keys_externally = provide_keys_externally

        # Project query vector to ``key_size`` dimension only if mismatched.
        self._query_projection = (
            nn.Linear(self.query_size, self.key_size, bias=False)
            if self.key_size != self.query_size
            else nn.Identity()
        )
        # Get key vectors from value vectors when not provided externally.
        # They are obtained by either projecting value vectors to ``key_size``
        # dimension. If value vectors are already of ``key_size`` dimension,
        # they are used as key vectors for the query.
        self._key_projection = (
            nn.Identity()
            if self._provide_keys_externally or self.key_size == self.value_size
            else nn.Linear(self.value_size, self.key_size, bias=False)
        )

    def forward(
        self,
        values: torch.Tensor,
        queries: torch.Tensor,
        keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Project query vector to ``key_size`` to compute attention weights.
        # There may be multiple queries per instance in a batch.
        # shape: (batch_size, num_queries, key_size)
        queries = self._query_projection(queries)
        if len(queries.size()) < 3:
            queries = queries.unsqueeze(1)

        if self._provide_keys_externally and keys is None:
            raise ValueError(
                f"{self.__class__.__name__} was constructed with "
                "`provide_keys_externally=True` but key vectors not provided."
            )

        # shape: (batch_size, num_candidates, key_size)
        keys = keys or values
        keys = self._key_projection(keys)

        # Scale by ``sqrt(key_size)`` to avoid softmax having small gradients.
        # shape: (batch_size, num_queries, num_candidates)
        attention_weights = torch.softmax(
            queries.matmul(keys.transpose(-1, -2)) / keys.size(-1) ** 0.5,
            dim=-1
        )
        # shape: (batch_size, num_queries, value_size)
        attended_values = attention_weights.matmul(values)
        return attended_values
