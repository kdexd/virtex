import torch
from torch import nn


class Fusion(nn.Module):
    r"""
    A base class for performing fusion between separate visual and textual
    modalities. All classes inheriting this should implement the fusion
    mechanism by overriding :meth:`forward` method, and specify the output
    size of fused vector by setting :py:attr:`~fused_feature_size`.

    Parameters
    ----------
    feature_size: int
        Size of the visual and textual features to be fused (last dimension).
        You need to make sure both have same size (they can be projected to a
        common size using :class:`torch.nn.Linear` outside this class).
    """

    def __init__(self, feature_size: int):
        super().__init__()
        self.feature_size = feature_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Fuse the given visual and textual features into one.

        Parameters
        ----------
        visual_features: torch.Tensor
            Features from last stage of a CNN (optionally projected to a common
            ``feature_size``). These are not globally average pooled. A tensor
            of shape ``(batch_size, spatial_size, spatial_size, feature_size)``.
        textual_features: torch.Tensor
            Features for each caption token (optionally projected to a common
            ``feature_size``). Operations at padded positions happen, and
            should be masked/ignored outside this class appropriately. A tensor
            of shape ``(batch_size, num_caption_tokens, feature_size)``.

        Returns
        -------
        torch.Tensor
            Fused feature vector for each caption token position. Typically a
            tensor of shape ``(batch_size, num_caption_tokens, fused_feature_size)``.
        """
        raise NotImplementedError

    @property
    def fused_feature_size(self):
        r"""
        Size of last dimension of fused features. Typically it will be
        ``feature_size`` or ``2 * feature_size``. All inherited classes
        must set this property.
        """
        raise NotImplementedError


class NoFusion(Fusion):
    r"""
    Perform no fusion between visual and textual features. Given our pretext
    tasks are all about producing a distribution over vocabulary tokens, this
    class ignores the visual features and only returns textual features, so it
    is _almost_ a no-op ("almost" because we throw away visual features).
    """

    def __init__(self, feature_size: int, *args, **kwargs):
        # *args and **kwargs for switching calls without changing signature.
        super().__init__(feature_size)

    @property
    def fused_feature_size(self):
        return self.feature_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:
        # shape: (batch_size, num_caption_tokens, feature_size)
        return textual_features


class ConcatenateFusion(Fusion):
    r"""
    Fuse visual and textual features by concatenating them. Visual features are
    global average pooled, and repeated for each caption token before fusion.
    Fused features undergo layer normalization and optionally dropout.

    Size of fused vector will be ``2 * feature_size``.
    """

    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.1,
    ):
        super().__init__(feature_size)
        # Do normalization of fused features. This helps a bit.
        self.layer_norm = nn.LayerNorm(
            self.fused_feature_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    @property
    def fused_feature_size(self):
        return 2 * self.feature_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:
        # Visual features will be grid features from CNN, perform global
        # average pooling.
        # shape: (batch_size, feature_size)
        visual_features = torch.mean(visual_features, dim=1)

        # Repeat image features for each token position in caption (padding
        # tokens will be ignored later anyway).
        batch_size, num_caption_tokens, _ = textual_features.size()
        # shape: (batch_size, num_caption_tokens, feature_size)
        visual_features = visual_features.unsqueeze(1).repeat(
            1, num_caption_tokens, 1
        )
        # shape: (batch_size, num_caption_tokens, 2 * feature_size)
        fused_features = torch.cat((visual_features, textual_features), dim=2)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features


class ElementwiseFusion(Fusion):
    r"""
    Fuse visual and textual features by element-wise operation. The
    operations supported are additive and multiplicative. Visual features are
    global average pooled, and repeated for each caption token before fusion.
    Fused features undergo layer normalization and optionally dropout.

    Size of fused vector will be ``feature_size``.
    """

    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.1,
        operation: str = "multiplicative",
    ):
        super().__init__(feature_size)
        if operation not in {"additive", "multiplicative"}:
            raise ValueError(
                "Supported ops in ElementwiseFusion: additive, multiplicative."
                f"Found {operation}."
            )
        self._operation = operation

        # Do normalization of fused features. This helps a bit.
        self.layer_norm = nn.LayerNorm(
            self.fused_feature_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    @property
    def fused_feature_size(self):
        return self.feature_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:

        # Visual features will be grid features from CNN, perform global
        # average pooling.
        # shape: (batch_size, feature_size)
        visual_features = torch.mean(visual_features, dim=1)

        # shape: (batch_size, num_caption_tokens, feature_size)
        if self._operation == "additive":
            fused_features = visual_features.unsqueeze(1) + textual_features
        else:
            fused_features = visual_features.unsqueeze(1) * textual_features

        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features


class MultiheadAttentionFusion(Fusion):
    r"""
    Fuse visual and textual features by pooling spatial visual features using
    :class:`~torch.nn.modules.activation.MultiheadAttention` (query is textual
    features). Attended visual features are repeated for each caption token,
    and concatenated with textual features. Fused features undergo layer
    normalization and optionally dropout.

    Size of fused vector will be ``2 * feature_size``.
    """

    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.1,
        attention_heads: int = 8,
    ):
        super().__init__(feature_size)
        self.attention = nn.MultiheadAttention(
            self.feature_size, attention_heads, dropout=0.1
        )
        # Do normalization of fused features. This helps a bit.
        self.layer_norm = nn.LayerNorm(
            self.fused_feature_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    @property
    def fused_feature_size(self):
        return 2 * self.feature_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:

        # Transpose batches as per required input format to multihead attention.
        # shape: (num_queries/keys/values, batch_size, feature_size)
        visual_features = visual_features.transpose(0, 1)
        textual_features = textual_features.transpose(0, 1)

        # Textual features are query, and visual features are keys and values.
        attended_features, _ = self.attention(
            textual_features, visual_features, visual_features
        )
        # Bring back batch_size to zero-th dimension.
        attended_features = attended_features.transpose(0, 1)
        textual_features = textual_features.transpose(0, 1)

        # shape: (batch_size, num_caption_tokens, 2 * feature_size)
        fused_features = torch.cat((attended_features, textual_features), dim=2)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features
