from typing import Optional, Tuple

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
    visual_feature_size: int
        Size of the visual features (last dimension). This is commonly the
        number of channels in final conv layer for ResNet-like models.
    textual_feature_size: int
        Size of the textual feature (last dimension). This is commonly the
        hidden size in Transformer-like models.
    projection_size: int
        Common size to which both features should be projected before fusion.
    """

    def __init__(
        self,
        visual_feature_size: int,
        textual_feature_size: int,
        projection_size: Optional[int] = None,
    ):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.textual_feature_size = textual_feature_size
        self.projection_size = projection_size or textual_feature_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Fuse the given visual and textual features into one.

        Parameters
        ----------
        visual_features: torch.Tensor
            Features from last stage of a CNN. A tensor of shape
            ``(batch_size, spatial_size, spatial_size, visual_feature_size)``.
            These are not globally average pooled.
        textual_features: torch.Tensor
            Features for each caption token. A tensor of shape
            ``(batch_size, num_caption_tokens, textual_feature_size)``.
            Operations at padded positions happen, and should be masked/ignored
            outside this class appropriately.

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
        ``projection_size`` or ``2 * projection_size``. All inherited classes
        must set this.
        """
        raise NotImplementedError


class ConcatenateFusion(Fusion):
    r"""
    Fuse visual and textual features by concatenating them. Visual features are
    global average pooled, and repeated for each caption token before fusion.
    Fused features undergo layer normalization and optionally dropout.

    Size of fused vector will be ``2 * projection_size``.
    """

    def __init__(
        self,
        visual_feature_size: int,
        textual_feature_size: int,
        projection_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__(visual_feature_size, textual_feature_size, projection_size)
        # Fully connected layers for projecting both modalities to a common
        # size before concatenation.
        self.projections = _VisualAndTextualProjections(
            self.visual_feature_size, self.textual_feature_size, self.projection_size
        )
        # Do normalization of fused features. This helps a bit.
        self.layer_norm = nn.LayerNorm(
            self.fused_feature_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    @property
    def fused_feature_size(self):
        return 2 * self.projection_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:
        # Visual features will be grid features from CNN, perform global
        # average pooling.
        # shape: (batch_size, visual_feature_size)
        visual_features = torch.mean(visual_features, dim=1)

        # Project visual and textual features to common size.
        visual_features, textual_features = self.projections(
            visual_features, textual_features
        )
        # Repeat image features for each token position in caption (padding
        # tokens will be ignored later anyway).
        batch_size, num_caption_tokens, _ = textual_features.size()
        # shape: (batch_size, num_caption_tokens, visual_feature_size)
        visual_features = visual_features.unsqueeze(1).repeat(
            1, num_caption_tokens, 1
        )
        # shape: (batch_size, num_caption_tokens, 2 * projection_size)
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

    Size of fused vector will be ``projection_size``.
    """

    def __init__(
        self,
        visual_feature_size: int,
        textual_feature_size: int,
        projection_size: Optional[int] = None,
        dropout: float = 0.1,
        operation: str = "multiplicative",
    ):
        super().__init__(visual_feature_size, textual_feature_size, projection_size)
        if operation not in {"additive", "multiplicative"}:
            raise ValueError(
                "Supported ops in ElementwiseFusion: additive, multiplicative."
                f"Found {operation}."
            )
        self._operation = operation

        # Fully connected layers for projecting both modalities to a common
        # size before fusion.
        self.projections = _VisualAndTextualProjections(
            self.visual_feature_size, self.textual_feature_size, self.projection_size
        )
        # Do normalization of fused features. This helps a bit.
        self.layer_norm = nn.LayerNorm(
            self.fused_feature_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    @property
    def fused_feature_size(self):
        return self.projection_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:

        # Visual features will be grid features from CNN, perform global
        # average pooling.
        # shape: (batch_size, visual_feature_size)
        visual_features = torch.mean(visual_features, dim=1)

        # Project visual and textual features to common size.
        visual_features, textual_features = self.projections(
            visual_features, textual_features
        )
        # Repeat image features for each token position in caption (padding
        # tokens will be ignored later anyway).
        batch_size, num_caption_tokens, _ = textual_features.size()
        # shape: (batch_size, num_caption_tokens, visual_feature_size)
        visual_features = visual_features.unsqueeze(1).repeat(
            1, num_caption_tokens, 1
        )
        # shape: (batch_size, projection_size)
        if self._operation == "additive":
            fused_features = visual_features + textual_features
        else:
            fused_features = visual_features * textual_features

        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features


class MultiheadAttentionFusion(Fusion):
    r"""
    Fuse visual and textual features by pooling spatial visual features using
    :class:`~torch.nn.modules.activation.MultiheadAttention` using textual
    features. Attended visual features are repeated for each caption token,
    and concatenated with textual features. Fused features undergo layer
    normalization and optionally dropout.

    Size of fused vector will be ``2 * projection_size``.
    """

    def __init__(
        self,
        visual_feature_size: int,
        textual_feature_size: int,
        projection_size: Optional[int] = None,
        dropout: float = 0.1,
        attention_heads: int = 8,
    ):
        super().__init__(visual_feature_size, textual_feature_size, projection_size)
        # Fully connected layers for projecting both modalities to a common
        # size before multihead attention.
        self.projections = _VisualAndTextualProjections(
            self.visual_feature_size, self.textual_feature_size, self.projection_size
        )
        self.attention = nn.MultiheadAttention(
            self.projection_size, attention_heads, dropout=0.1
        )
        # Do normalization of fused features. This helps a bit.
        self.layer_norm = nn.LayerNorm(
            self.fused_feature_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    @property
    def fused_feature_size(self):
        return 2 * self.projection_size

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> torch.Tensor:

        # Project visual and textual features to common size.
        visual_features, textual_features = self.projections(
            visual_features, textual_features
        )
        # Transpose batches as per required input format to multihead attention.
        # shape: (num_queries/keys/values, batch_size, projection_size)
        visual_features = visual_features.transpose(0, 1)
        textual_features = textual_features.transpose(0, 1)

        # Textual features are query, and visual features are keys and values.
        attended_features, _ = self.attention(
            textual_features, visual_features, visual_features
        )
        # Bring back batch_size to zero-th dimension.
        attended_features = attended_features.transpose(0, 1)
        textual_features = textual_features.transpose(0, 1)

        # shape: (batch_size, num_caption_tokens, 2 * projection_size)
        fused_features = torch.cat((attended_features, textual_features), dim=2)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features


class _VisualAndTextualProjections(nn.Module):
    r"""
    A simple module with two :class:`torch.nn.Linear` layers to project visual
    and textual features to the same size (separately). It is commonly used in
    many classes of this module (_only_).

    Parameters
    ----------
    visual_feature_size: int
        Size of the visual features (last dimension). This is commonly the
        number of channels in final conv layer for ResNet-like models.
    textual_feature_size: int
        Size of the textual feature (last dimension). This is commonly the
        hidden size in Transformer-like models.
    projection_size: int
        Common size to which both features should be projected.
    """

    def __init__(
        self,
        visual_feature_size: int,
        textual_feature_size: int,
        projection_size: int,
    ):
        super().__init__()
        self._v_projection = (
            nn.Linear(visual_feature_size, projection_size, bias=False)
            if visual_feature_size != projection_size
            else nn.Identity()
        )
        self._t_projection = (
            nn.Linear(textual_feature_size, projection_size, bias=False)
            if textual_feature_size != projection_size
            else nn.Identity()
        )
    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Given visual and textual features, project to a common size separately.

        Parameters
        ----------
        visual_features: torch.Tensor
            A tensor of shape ``(batch_size, ..., visual_feature_size)``.
        textual_features: torch.Tensor
            A tensor of shape ``(batch_size, ..., textual_feature_size)``.

        Returns
        -------
        torch.Tensor, torch.Tensor
            A tuple of projected visual and textual features, both of size
            ``(batch_size, ..., projection_size)``.
        """
        # shape: (batch_size, projection_size)
        visual_features = self._v_projection(visual_features)
        textual_features = self._t_projection(textual_features)

        return visual_features, textual_features
