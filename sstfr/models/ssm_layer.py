"""
Complex-valued diagonal SSM layer with Gabor-consistent exponential parameterization.

Implements equations (1)-(4) of the SSTFR paper. For channel d, the hidden state
evolves as:

    h_t^(d) = a_d * h_{t-1}^(d) + b_d * x_t

with a_d = exp(alpha_d + i*omega_d), alpha_d < 0, omega_d in R.

Under this parameterization, the unrolled impulse response is a causal,
exponentially-decaying complex sinusoid (a "one-sided Gabor atom"):

    g_d(tau) = b_d * exp((alpha_d + i*omega_d) * tau)   for tau >= 0.

This is verified numerically by tests/test_gabor_equivalence.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SSTFRConfig:
    """Configuration for the SSTFR front-end.

    Attributes:
        num_channels: Number of complex SSM channels (D in the paper). Default 128.
        sample_rate: Audio sample rate in Hz (f_s in the paper).
        f_min: Minimum center frequency in Hz for Mel-spaced initialization.
        f_max: Maximum center frequency in Hz. Defaults to sample_rate / 2.
        window_samples: Effective window length T_w * f_s (in samples) used to
            initialize alpha_d = -c / window_samples. Controls initial bandwidth.
        decay_c: Constant c > 0 in alpha_d = -c / window_samples. Larger c -> faster
            decay -> wider bandwidth.
        learn_alpha: If True, alpha_d is learnable. If False, it is frozen at init.
        learn_omega: If True, omega_d is learnable.
        learn_b: If True, b_d (magnitude and phase) is learnable.
    """

    num_channels: int = 128
    sample_rate: int = 16000
    f_min: float = 40.0
    f_max: float | None = None
    window_samples: int = 400  # ~25ms at 16kHz, similar to STFT
    decay_c: float = 4.0
    learn_alpha: bool = True
    learn_omega: bool = True
    learn_b: bool = True


def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    """Hz to Mel (Slaney formula, standard in torchaudio/librosa)."""
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    """Mel to Hz (inverse Slaney)."""
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_spaced_frequencies(num: int, f_min: float, f_max: float) -> torch.Tensor:
    """Return `num` frequencies in Hz, spaced linearly on the Mel scale."""
    m_min = _hz_to_mel(torch.tensor(f_min))
    m_max = _hz_to_mel(torch.tensor(f_max))
    mels = torch.linspace(m_min.item(), m_max.item(), num)
    return _mel_to_hz(mels)


class SSTFRLayer(nn.Module):
    """Complex diagonal SSM as a Gabor-like filter bank.

    Input:  real tensor of shape (B, L) -- audio waveform
    Output: complex tensor of shape (B, L, D) -- hidden states H

    The forward pass uses a naive O(L) recurrence, which is correct and simple.
    For long sequences we can swap in a parallel scan later (the recurrence is a
    linear first-order IIR and admits an efficient parallel prefix-sum
    implementation). Correctness first, speed second.
    """

    def __init__(self, config: SSTFRConfig):
        super().__init__()
        self.config = config

        D = config.num_channels
        f_max = config.f_max if config.f_max is not None else config.sample_rate / 2

        # --- omega_d: center angular frequency, normalized (radians/sample) ---
        # omega_d = 2*pi * f_d / f_s. Mel-spaced init between f_min and f_max.
        freqs_hz = mel_spaced_frequencies(D, config.f_min, f_max)
        omega_init = 2.0 * math.pi * freqs_hz / config.sample_rate  # shape (D,)
        self.omega = nn.Parameter(omega_init, requires_grad=config.learn_omega)

        # --- alpha_d: negative real scalar per channel ---
        # We parameterize alpha_d = -softplus(alpha_raw) so alpha_d < 0 is
        # guaranteed and the parameter alpha_raw is unconstrained (stable gradients).
        # Init: alpha = -c / window_samples  =>  softplus(alpha_raw) = c / window_samples
        # So alpha_raw = log(exp(c / window_samples) - 1).
        alpha_magnitude = config.decay_c / config.window_samples  # positive scalar
        alpha_raw_init = math.log(math.expm1(alpha_magnitude))
        self.alpha_raw = nn.Parameter(
            torch.full((D,), alpha_raw_init), requires_grad=config.learn_alpha
        )

        # --- b_d: complex input projection, stored as (log|b|, phase) ---
        # Initial magnitude chosen so the peak response is O(1). A standard choice
        # for a causal exponential filter is |b| = sqrt(1 - exp(2*alpha)) so that
        # the L2 norm of the impulse response is 1. We use a simpler init here
        # (|b| = sqrt(-2*alpha)) which also yields unit L2 norm in the limit.
        b_mag_init = torch.sqrt(torch.tensor(2.0 * alpha_magnitude))  # scalar
        self.b_log_mag = nn.Parameter(
            torch.full((D,), math.log(b_mag_init.item())), requires_grad=config.learn_b
        )
        # Random phase to break channel symmetry (crucial when b is shared across
        # channels of identical |a|). Deterministic via torch default generator.
        phase_init = 2.0 * math.pi * torch.rand(D)
        self.b_phase = nn.Parameter(phase_init, requires_grad=config.learn_b)

    # ------------------------------------------------------------------
    # Derived parameters (used both internally and by the alignment loss)
    # ------------------------------------------------------------------

    def alpha(self) -> torch.Tensor:
        """alpha_d in R_<0. Shape (D,)."""
        return -torch.nn.functional.softplus(self.alpha_raw)

    def b(self) -> torch.Tensor:
        """Complex input projection b_d. Shape (D,), dtype complex64."""
        mag = torch.exp(self.b_log_mag)
        return torch.polar(mag, self.b_phase)

    def a(self) -> torch.Tensor:
        """Complex pole a_d = exp(alpha_d + i*omega_d). Shape (D,), complex64."""
        return torch.exp(torch.complex(self.alpha(), self.omega))

    def center_frequencies_hz(self) -> torch.Tensor:
        """Current center frequencies in Hz. Useful for analysis/plots."""
        return self.omega * self.config.sample_rate / (2.0 * math.pi)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_power: bool = False,
    ) -> torch.Tensor:
        """Run the SSM on a batch of waveforms via pure real-arithmetic FIR conv.

        This rewrite eliminates all complex-dtype tensors from the hot path.
        Using the decomposition

            a_d^tau = exp(alpha_d * tau) * [cos(omega_d * tau) + i*sin(omega_d * tau)]

        and b_d = |b_d| * exp(i * phi_d), the impulse response g_d(tau) splits
        into real and imaginary parts:

            Re(g_d(tau)) = |b_d| * exp(alpha_d * tau) * cos(phi_d + omega_d * tau)
            Im(g_d(tau)) = |b_d| * exp(alpha_d * tau) * sin(phi_d + omega_d * tau)

        Convolution of the real input x with Re(g) and Im(g) gives the real and
        imaginary parts of H. This is mathematically identical to the complex
        formulation (Theorem 1) but never instantiates a complex tensor. Every
        autograd op is on real tensors, which map to highly optimized CUDA
        kernels on every platform.

        Args:
            x: real tensor of shape (B, L).
            return_power: If True, return |H|^2 as a real tensor (B, L, D).
                If False (default), return complex H (B, L, D) by combining
                the real and imaginary conv outputs at the last moment. Only
                the return_power=False path pays the complex-dtype cost, and
                only at the final output -- the training-heavy path uses
                return_power=True and stays fully real.

        Returns:
            complex (B, L, D) if return_power=False, else real (B, L, D).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, L); got {tuple(x.shape)}.")

        B, L = x.shape
        D = self.config.num_channels
        device = x.device

        # --- Derive real parameters ---
        alpha = self.alpha()                    # (D,) real, < 0
        omega = self.omega                      # (D,) real
        b_mag = torch.exp(self.b_log_mag)       # (D,) real, positive
        phi = self.b_phase                      # (D,) real

        # --- Adaptive kernel length K (quantized for cuDNN cache) ---
        kernel_tol = 1e-5
        with torch.no_grad():
            min_abs_alpha_val = float(alpha.abs().min().cpu().item())
        if not math.isfinite(min_abs_alpha_val) or min_abs_alpha_val < 1e-8:
            K = L
        else:
            K = int(math.ceil(math.log(1.0 / kernel_tol) / min_abs_alpha_val))
        K = max(64, min(K, L))
        K = ((K + 63) // 64) * 64  # quantize to multiple of 64
        K = min(K, L)

        # --- Build kernel in pure real arithmetic ---
        # tau: (K,), phase arg: (K, D), broadcast over channels.
        tau = torch.arange(K, device=device, dtype=alpha.dtype)

        # Envelope: exp(alpha * tau) -- real, shape (K, D)
        envelope = torch.exp(alpha.unsqueeze(0) * tau.unsqueeze(1))

        # Phase argument: phi + omega * tau -- real, shape (K, D)
        phase_arg = phi.unsqueeze(0) + omega.unsqueeze(0) * tau.unsqueeze(1)

        # Real and imaginary parts of b * a^tau, shape (K, D)
        scale = b_mag.unsqueeze(0) * envelope      # (K, D) amplitude envelope
        g_real = scale * torch.cos(phase_arg)      # (K, D)
        g_imag = scale * torch.sin(phase_arg)      # (K, D)

        # --- Fused real conv1d ---
        # Flip kernels for causal convolution (conv1d is cross-correlation).
        g_real_flipped = g_real.flip(0)            # (K, D)
        g_imag_flipped = g_imag.flip(0)            # (K, D)

        # Stack as (2D, 1, K) for a single conv1d call.
        weight = torch.stack(
            [
                g_real_flipped.transpose(0, 1),    # (D, K)
                g_imag_flipped.transpose(0, 1),    # (D, K)
            ],
            dim=0,
        )                                          # (2, D, K)
        weight = weight.reshape(2 * D, 1, K).contiguous()

        # Input: (B, 1, L) float32, left-padded with K-1 zeros for causal conv.
        x_in = x.unsqueeze(1)                       # (B, 1, L)
        x_padded = torch.nn.functional.pad(x_in, (K - 1, 0))  # (B, 1, L + K - 1)

        # Single real conv1d: output (B, 2D, L). First D channels = real, next D = imag.
        H_stacked = torch.nn.functional.conv1d(x_padded, weight)  # (B, 2D, L)
        H_real_out = H_stacked[:, :D, :]            # (B, D, L) real
        H_imag_out = H_stacked[:, D:, :]            # (B, D, L) real

        if return_power:
            # |H|^2 = Re(H)^2 + Im(H)^2, entirely real.
            power = H_real_out.pow(2) + H_imag_out.pow(2)  # (B, D, L) real
            return power.transpose(1, 2).contiguous()      # (B, L, D) real

        # Only build complex tensor if the caller explicitly needs it.
        H = torch.complex(H_real_out, H_imag_out).transpose(1, 2).contiguous()
        return H

__all__ = ["SSTFRConfig", "SSTFRLayer", "mel_spaced_frequencies"]
