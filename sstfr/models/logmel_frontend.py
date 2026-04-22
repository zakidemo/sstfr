"""
Log-Mel spectrogram baseline front-end.

Wraps torchaudio.transforms.MelSpectrogram so it has the same interface as our
SSTFR and LEAF front-ends: input (B, L) real audio, output (B, D, T) real.

We do not reinvent the Mel spectrogram -- torchaudio's implementation is the
reference in the audio community. The purpose of this module is only to
standardize the interface and the log-compression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelFrontend(nn.Module):
    """Fixed (non-learnable) log-Mel spectrogram front-end.

    Args:
        sample_rate: Audio sample rate (Hz).
        n_mels: Number of Mel bands (D). Default 128 to match SSTFR's D.
        n_fft: FFT size. Default 1024 (~64 ms at 16 kHz, similar to SSTFR T_w).
        hop_length: Hop size in samples. Default 160 (10 ms at 16 kHz).
        f_min / f_max: Frequency range. Default 40 Hz to sample_rate/2.
        log_eps: Floor for log to avoid log(0).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        f_min: float = 40.0,
        f_max: float | None = None,
        log_eps: float = 1e-6,
    ):
        super().__init__()
        if f_max is None:
            f_max = sample_rate / 2

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.log_eps = log_eps

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,  # magnitude squared
        )

    @property
    def output_channels(self) -> int:
        """Number of frequency bins in the output (D)."""
        return self.n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-Mel spectrogram.

        Args:
            x: real tensor of shape (B, L).

        Returns:
            features: real tensor of shape (B, D, T) where D = n_mels and
                T = L // hop_length + 1.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected (B, L); got {tuple(x.shape)}")

        mel = self.mel(x)  # (B, D, T), power-magnitude
        log_mel = torch.log(mel + self.log_eps)
        return log_mel


__all__ = ["LogMelFrontend"]
