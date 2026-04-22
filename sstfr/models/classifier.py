"""
ResNet-18 classifier head for spectrogram-like inputs.

This is the "downstream task head" that all front-ends feed into. Reviewers
asked for a high-capacity backend so any accuracy difference isolates the
front-end. ResNet-18 is standard in audio classification literature.

We take the torchvision ResNet-18 and modify the first conv layer to accept
a single-channel (1, D, T) input instead of the default 3-channel RGB image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet18Head(nn.Module):
    """ResNet-18 for single-channel (B, D, T) spectrogram input.

    Args:
        num_classes: Number of output classes (e.g., 50 for ESC-50).
        pretrained: If True, load ImageNet pre-trained weights and adapt the
            first conv. Default False (train from scratch) to match the paper.
    """

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Modify first conv: 3 channels -> 1 channel.
        # Standard trick: average the 3 input kernels into 1.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # Replace the final FC layer
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        self.backbone = backbone

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: real tensor of shape (B, D, T). Will be unsqueezed to
                (B, 1, D, T) for conv2d.

        Returns:
            logits: tensor of shape (B, num_classes).
        """
        if features.dim() != 3:
            raise ValueError(f"Expected (B, D, T); got {tuple(features.shape)}")
        x = features.unsqueeze(1)  # (B, 1, D, T)
        return self.backbone(x)


__all__ = ["ResNet18Head"]
