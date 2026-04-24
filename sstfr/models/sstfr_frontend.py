"""
SSTFR front-end wrapper: SSM layer -> power -> log -> pool -> (B, D, T).

The SSTFRLayer produces complex hidden states H of shape (B, L, D) at the
audio sample rate. For the downstream classifier we need a real feature map of
shape (B, D, T) with T much smaller than L (otherwise ResNet would have to
process 80k timesteps).

Pipeline:
  H (B, L, D) complex  OR  |H|^2 (B, L, D) real  (dual-path; see below)
    -> log(. + eps)
    -> avg-pool over `hop_length` along the time axis
    -> transpose to (B, D, T)

Dual-path design (for WSL2 speed):
  The complex dtype triggers slow kernels on WSL2 (approx 50x slowdown for
  conv1d and abs on complex64). When the alignment loss is not needed, we
  compute |H|^2 directly from the SSM layer's real/imag conv1d outputs,
  skipping the torch.complex() construction entirely. This is controlled
  by the `_need_complex_H` flag set via `set_need_complex_H(bool)`.
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
        # Controls whether forward() caches complex H for the alignment loss.
        # Default False -> fast real-tensor path. The trainer sets True
        # during SSTFR+L_SSA training steps.
        self._need_complex_H: bool = False
        self._last_H: torch.Tensor | None = None

    @property
    def output_channels(self) -> int:
        return self.ssm.config.num_channels

    def set_need_complex_H(self, need: bool) -> None:
        """Tell the frontend whether the next forward pass needs to produce
        complex H (for the alignment loss) or can use the fast real path.
        Call this from the training loop before each forward.
        """
        self._need_complex_H = need

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SSTFR feature map.

        Args:
            x: real tensor of shape (B, L).

        Returns:
            features: real tensor of shape (B, D, T).
        """
        # Two forward paths:
        #   - If the alignment loss is NOT needed this step, use the fast
        #     real-tensor path (return_power=True). This avoids complex dtype
        #     construction and abs(), both pathologically slow on WSL2.
        #   - If we need H for L_SSA (training + alignment enabled), use the
        #     standard complex path.
        if self._need_complex_H and self.training:
            H = self.ssm(x, return_power=False)  # (B, L, D) complex
            self._last_H = H
            power = H.abs().pow(2)  # (B, L, D) real
        else:
            # Fast path: get |H|^2 directly from the SSM layer.
            power = self.ssm(x, return_power=True)  # (B, L, D) real
            self._last_H = None

        log_power = torch.log(power + self.log_eps)  # (B, L, D)

        # Average pool along time to downsample to (B, D, T).
        # Transpose to (B, D, L), pool on L, keep shape (B, D, T).
        log_power_bdl = log_power.transpose(1, 2)  # (B, D, L)
        pooled = F.avg_pool1d(
            log_power_bdl, kernel_size=self.hop_length, stride=self.hop_length
        )
        return pooled  # (B, D, T)

    @property
    def last_hidden_states(self) -> torch.Tensor | None:
        """The last complex H tensor from the SSTFR layer, if cached.
        Only populated when forward was called with need_complex_H=True and
        the model was in training mode.
        """
        return self._last_H


__all__ = ["SSTFRFrontend"]
