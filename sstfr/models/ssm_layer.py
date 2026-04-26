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

    Forward pass uses FFT-based convolution. See `forward()` docstring for
    the rationale (cuDNN's algorithm selection is poor for very-long-kernel
    grouped conv on the input shapes used here).
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
        """Run the SSM on a batch of waveforms via FFT-based convolution.

        Mathematically identical to the recurrence h_t = a*h_{t-1} + b*x_t
        with h_0 = 0, evaluated as the convolution of x with the truncated
        impulse response g_d(tau) = b_d * a_d^tau (tau = 0, ..., K-1).

        Implementation note: we use FFT-based convolution rather than
        torch.nn.functional.conv1d because cuDNN's algorithm selection for
        very-long-kernel grouped conv (here, out_channels=2D=256, kernel
        length ~1152, input length ~80000) is poor -- profiling showed
        ~700 ms/iter at B=4 vs ~50 ms/iter for FFT conv on an RTX 4070
        Laptop. The FFT path is also more numerically accurate (fewer
        accumulated multiply-adds).

        Using the decomposition

            a_d^tau = exp(alpha_d * tau) * [cos(omega_d * tau) + i*sin(omega_d * tau)]

        and b_d = |b_d| * exp(i * phi_d), the impulse response splits into:

            Re(g_d(tau)) = |b_d| * exp(alpha_d * tau) * cos(phi_d + omega_d * tau)
            Im(g_d(tau)) = |b_d| * exp(alpha_d * tau) * sin(phi_d + omega_d * tau)

        We FFT both kernels and the input, multiply in the frequency domain,
        and inverse-FFT to recover the real and imaginary parts of the output.
        No complex dtype is used until the final step (and only if requested).

        Args:
            x: real tensor of shape (B, L).
            return_power: If True, return |H|^2 as a real tensor (B, L, D).
                If False (default), return complex H (B, L, D).

        Returns:
            complex (B, L, D) if return_power=False, else real (B, L, D).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, L); got {tuple(x.shape)}.")

        B, L = x.shape
        D = self.config.num_channels
        device = x.device
        dtype = x.dtype

        # --- Derive real parameters ---
        alpha = self.alpha()                    # (D,) real, < 0
        omega = self.omega                      # (D,) real
        b_mag = torch.exp(self.b_log_mag)       # (D,) real, positive
        phi = self.b_phase                      # (D,) real

        # --- Adaptive kernel length K ---
        # Truncate the impulse response where exp(alpha * K) < kernel_tol.
        # Quantize to multiples of 64 so K stays stable across small alpha
        # drifts during training (avoids re-planning the FFT every step).
        kernel_tol = 1e-5
        with torch.no_grad():
            min_abs_alpha_val = float(alpha.abs().min().cpu().item())
        if not math.isfinite(min_abs_alpha_val) or min_abs_alpha_val < 1e-8:
            K = L
        else:
            K = int(math.ceil(math.log(1.0 / kernel_tol) / min_abs_alpha_val))
        K = max(64, min(K, L))
        K = ((K + 63) // 64) * 64
        K = min(K, L)

        # --- Build kernels in pure real arithmetic ---
        tau = torch.arange(K, device=device, dtype=dtype)
        envelope = torch.exp(alpha.unsqueeze(0) * tau.unsqueeze(1))  # (K, D)
        phase_arg = phi.unsqueeze(0) + omega.unsqueeze(0) * tau.unsqueeze(1)
        scale = b_mag.unsqueeze(0) * envelope                        # (K, D)
        g_real = (scale * torch.cos(phase_arg)).transpose(0, 1).contiguous()  # (D, K)
        g_imag = (scale * torch.sin(phase_arg)).transpose(0, 1).contiguous()  # (D, K)

        # --- FFT-based convolution ---
        # We need a transform length >= L + K - 1 to avoid circular wraparound.
        # Round up to the next power of two for cuFFT performance.
        n_fft = 1
        target = L + K - 1
        while n_fft < target:
            n_fft *= 2

        # Forward FFTs. Kernels are zero-padded to n_fft implicitly by rfft's
        # `n` argument; the input is similarly zero-padded.
        G_real = torch.fft.rfft(g_real, n=n_fft)  # (D, n_fft//2 + 1) complex
        G_imag = torch.fft.rfft(g_imag, n=n_fft)  # (D, n_fft//2 + 1) complex
        X = torch.fft.rfft(x, n=n_fft)            # (B, n_fft//2 + 1) complex

        # Pointwise multiplication in frequency domain, then inverse FFT.
        # Broadcasting: X is (B, F), kernels are (D, F). Unsqueeze to (B, 1, F)
        # and (1, D, F) so the product is (B, D, F).
        Xb = X.unsqueeze(1)                       # (B, 1, F)
        Y_real_full = torch.fft.irfft(Xb * G_real.unsqueeze(0), n=n_fft)  # (B, D, n_fft)
        Y_imag_full = torch.fft.irfft(Xb * G_imag.unsqueeze(0), n=n_fft)  # (B, D, n_fft)

        # Take the first L samples (causal output).
        H_real_out = Y_real_full[:, :, :L]        # (B, D, L)
        H_imag_out = Y_imag_full[:, :, :L]        # (B, D, L)

        if return_power:
            power = H_real_out.pow(2) + H_imag_out.pow(2)  # (B, D, L)
            return power.transpose(1, 2).contiguous()       # (B, L, D)

        H = torch.complex(H_real_out, H_imag_out).transpose(1, 2).contiguous()
        return H


__all__ = ["SSTFRConfig", "SSTFRLayer", "mel_spaced_frequencies"]
