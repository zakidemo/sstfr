"""
SSTFR front-end wrapper: SSM layer -> power -> log -> pool -> (B, D, T).

The SSTFRLayer produces complex hidden states H of shape (B, L, D) at the
audio sample rate. For the downstream classifier we need a real feature map of
shape (B, D, T) with T much smaller than L (otherwise ResNet would have to
process 80k timesteps).

Pipeline:
  H (B, L, D) complex
    -> |H|^2         (B, L, D) real       power
    -> log(. + eps)  (B, L, D) real       log-compression (matches Log-Mel)
    -> avg-pool over a hop window of `hop_length` along the L axis
                    (B, T, D)             T = L // hop_length
    -> transpose    (B, D, T)             to match the (B, D, T) convention

We also expose the underlying SSTFRLayer so the training loop can compute the
alignment loss on the raw complex H.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm_layer import SSTFRConfig, SSTFRLayer


class SSTFRFrontend(nn.Module):
    """SSTFR front-end producing a (B, D, T) real feature map.

    Args:
        config: SSTFRConfig for the underlying SSM layer.
        hop_length: Temporal downsampling factor. Default 160 (10 ms at 16 kHz),
            matching the Log-Mel baseline.
        log_eps: Floor for log compression.
    """

    def __init__(
        self,
        config: SSTFRConfig,
        hop_length: int = 160,
        log_eps: float = 1e-6,
    ):
        super().__init__()
        self.ssm = SSTFRLayer(config)
        self.hop_length = hop_length
        self.log_eps = log_eps

    @property
    def output_channels(self) -> int:
        return self.ssm.config.num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SSTFR feature map.

        Args:
            x: real tensor of shape (B, L).

        Returns:
            features: real tensor of shape (B, D, T).
        """
        H = self.ssm(x)  # (B, L, D), complex
        # Cache the complex hidden states so the training loop can access them
        # for the alignment loss without a second forward pass.
        self._last_H = H

        power = H.abs().pow(2)  # (B, L, D) real
        log_power = torch.log(power + self.log_eps)  # (B, L, D)

        # Average pool along time to downsample to (B, T, D)
        # Transpose to (B, D, L), pool on L, transpose back.
        log_power_bdl = log_power.transpose(1, 2)  # (B, D, L)
        pooled = F.avg_pool1d(log_power_bdl, kernel_size=self.hop_length, stride=self.hop_length)
        # pooled: (B, D, T) where T = L // hop_length
        return pooled

    @property
    def last_hidden_states(self) -> torch.Tensor:
        """The last H tensor returned by the SSTFR layer. For alignment loss."""
        return self._last_H


__all__ = ["SSTFRFrontend"]
