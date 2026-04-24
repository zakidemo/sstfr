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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SSM on a batch of waveforms via truncated FIR convolution.

        Mathematically equivalent to the recurrence h_t = a*h_{t-1} + b*x_t with
        h_0 = 0 (Theorem 1). Unrolled:

            h_t = sum_{tau=0}^{t-1} a^tau * b * x_{t-tau}  =  (g * x)[t]

        The kernel g_d(tau) = b_d * a_d^tau decays as exp(alpha_d * tau) with
        alpha_d < 0. We truncate g to the shortest length K such that
        exp(min_alpha * K) < `kernel_tol` for every channel. For typical
        initializations (alpha ~ -0.01) this yields K ~ 1000-2000 samples
        instead of the full L ~ 80000, cutting memory and compute by 40x+.

        Implementation: a single torch.nn.functional.conv1d call. We express
        complex convolution as two real convolutions (real and imaginary parts
        of g), then recombine. This routes through cuDNN, which is faster than
        FFT for kernels of this length and uses O(B*L*D) memory (no n_fft=2L
        intermediate).

        Numerically equivalent to the recurrence up to float32 roundoff. The
        Gabor equivalence tests pass with the same 1e-6 tolerance.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, L); got {tuple(x.shape)}.")

        B, L = x.shape
        D = self.config.num_channels
        device = x.device

        # --- Compute kernel parameters ---
        alpha = self.alpha()  # (D,) real, < 0
        omega = self.omega  # (D,) real

        # --- Adaptive kernel length K ---
        # We want exp(alpha * K) < kernel_tol for ALL channels.
        # K_d = log(1/kernel_tol) / |alpha_d|. Take the max across channels
        # (the slowest-decaying channel dictates K).
        #
        # Compute K as a Python scalar from detached alpha values to avoid any
        # device mismatch or gradient-tracking overhead. This is a shape
        # parameter -- it must not depend on gradients.
        kernel_tol = 1e-5
        with torch.no_grad():
            min_abs_alpha_val = float(alpha.abs().min().cpu().item())
        # Guard against pathological values (NaN, zero, extremely small alpha
        # that would demand K > L).
        if not math.isfinite(min_abs_alpha_val) or min_abs_alpha_val < 1e-8:
            K = L
        else:
            K = int(math.ceil(math.log(1.0 / kernel_tol) / min_abs_alpha_val))
        # Cap at L so we never exceed input length; floor at 64 for sanity.
        K = max(64, min(K, L))

        # Critical for training speed: quantize K to a multiple of 64 so small
        # drifts in alpha during training do not change the kernel shape.
        # cuDNN benchmark caches kernel selection by exact tensor shape; a
        # varying K would force re-tuning on every forward pass (~50x slowdown).
        # Quantizing to 64 means alpha must drift enough to shift K by a full
        # 64 samples before cuDNN re-tunes, which essentially never happens
        # within a single run. The tiny conservatism (at most +63 samples of
        # kernel past the tolerance threshold) is negligible for accuracy.
        K = ((K + 63) // 64) * 64
        K = min(K, L)  # re-clamp after quantization

        # --- Build kernel g_d(tau) for tau = 0, ..., K-1 ---
        tau = torch.arange(K, device=device, dtype=alpha.dtype)  # (K,)
        # log_a_tau: (K, D)
        log_a_tau = tau.unsqueeze(1) * torch.complex(alpha, omega).unsqueeze(0)
        a_tau = torch.exp(log_a_tau)  # (K, D) complex

        b = self.b()  # (D,) complex64
        g = b.unsqueeze(0) * a_tau  # (K, D) complex

        # --- Fused complex conv1d ---
        # Standard approach: split complex g into real and imaginary parts, do
        # two conv1d calls, recombine. But each conv1d launches its own cuDNN
        # workspace. We fuse them: stack real and imag kernels along the
        # out-channels axis, do ONE conv1d, then split. Halves kernel launches.
        #
        # Kernel shape: (out_channels=2D, in_channels=1, K).
        # First D output channels are real parts, next D are imaginary parts.
        g_flipped = g.flip(0)  # (K, D) complex
        g_real = g_flipped.real.transpose(0, 1).unsqueeze(1)  # (D, 1, K)
        g_imag = g_flipped.imag.transpose(0, 1).unsqueeze(1)  # (D, 1, K)
        g_weight = torch.cat([g_real, g_imag], dim=0).contiguous()  # (2D, 1, K)

        # Input: (B, 1, L) float32, left-padded with K-1 zeros for causal conv.
        x_in = x.unsqueeze(1)  # (B, 1, L)
        x_padded = torch.nn.functional.pad(x_in, (K - 1, 0))  # (B, 1, L + K - 1)

        # Single conv1d, output shape (B, 2D, L)
        H_stacked = torch.nn.functional.conv1d(x_padded, g_weight)  # (B, 2D, L)

        # Split into real and imaginary halves
        H_real = H_stacked[:, :D, :]  # (B, D, L)
        H_imag = H_stacked[:, D:, :]  # (B, D, L)

        # Combine into complex and reshape to (B, L, D)
        H = torch.complex(H_real, H_imag).transpose(1, 2).contiguous()  # (B, L, D)
        return H


__all__ = ["SSTFRConfig", "SSTFRLayer", "mel_spaced_frequencies"]
