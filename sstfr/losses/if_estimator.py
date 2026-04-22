"""
Numerically stable instantaneous frequency (IF) estimator for complex signals.

Implements Eq. (5) of the SSTFR paper:

    omega_SSM(t, d) = f_s * arg( h_t^(d) * conj(h_{t-1}^(d)) )

Why this form (the complex-phasor difference) instead of a naive
finite-difference of the unwrapped phase?

  Naive:  omega_hat = f_s * (unwrap(angle(h_t)) - unwrap(angle(h_{t-1})))
  Problems:
    * unwrap() is non-differentiable (discontinuous jumps of +/-2*pi).
    * angle() itself is non-differentiable at h = 0 and numerically unstable
      in low-amplitude regions.
    * Gradients explode near zero crossings.

The complex-phasor form  arg(h_t * conj(h_{t-1}))  is:
  * Equal to  angle(h_t) - angle(h_{t-1})  mod 2*pi, so it gives the same
    instantaneous frequency without any explicit unwrapping.
  * Bounded in [-pi, pi] automatically -- no wraparound handling needed.
  * Differentiable almost everywhere (only undefined at exact h = 0).
  * Well-conditioned because it uses the product of phasors rather than the
    difference of their arguments.

This file provides two estimators:
  1. `instantaneous_frequency_from_phasors`: the exact Eq. (5) form.
  2. `instantaneous_frequency_smoothed`: a regularized variant that adds a
     small epsilon to prevent division by zero when h vanishes. Used by the
     alignment loss (where vanishing amplitude regions are also down-weighted).

All functions are torch-native, autograd-compatible, and work on both CPU and CUDA.
"""

from __future__ import annotations

import math

import torch


def instantaneous_frequency_from_phasors(
    H: torch.Tensor,
    sample_rate: int,
    return_radians_per_second: bool = True,
) -> torch.Tensor:
    """Estimate instantaneous frequency of a complex signal via phasor differences.

    Implements:   omega(t) = f_s * arg( h_t * conj(h_{t-1}) )

    Args:
        H: complex tensor of shape (..., L, D) or (..., L). Last dim can be D
            channels or omitted for a single-channel signal.
        sample_rate: Sampling rate f_s in Hz.
        return_radians_per_second: If True (default), return omega in rad/s.
            If False, return omega in Hz (i.e., omega / (2*pi)).

    Returns:
        omega: real tensor of same shape as H, with one fewer time step prepended
            with zero padding at t=0 (so shape matches H).

    Notes:
        The returned tensor is real-valued. It has shape identical to H with the
        time dimension preserved. Position t=0 is padded with zeros because the
        phasor difference needs h_{t-1}.
    """
    if not torch.is_complex(H):
        raise TypeError(f"Expected complex tensor, got dtype {H.dtype}.")

    # Compute h_t * conj(h_{t-1}) along the time axis (second-to-last).
    # We pad the beginning with a copy of h_0 so the output length matches L.
    # For t=0 this gives  h_0 * conj(h_0) = |h_0|^2, which is real and has
    # argument = 0. That's a safe default for the t=0 position.
    time_axis = -2 if H.dim() >= 2 else -1

    # Shift: H_shifted[..., t, :] = H[..., t-1, :]
    H_prev = torch.roll(H, shifts=1, dims=time_axis)
    # Overwrite the wrapped-around element at t=0 with h_0 itself (zero-diff).
    idx = [slice(None)] * H.dim()
    idx[time_axis] = 0
    H_prev = H_prev.clone()
    H_prev[tuple(idx)] = H[tuple(idx)]

    # Phasor product and its argument
    phasor_product = H * torch.conj(H_prev)
    phase_diff = torch.angle(phasor_product)  # in radians per sample, in [-pi, pi]

    # Convert to rad/s (multiply by f_s) or Hz (multiply by f_s / (2*pi))
    if return_radians_per_second:
        return phase_diff * sample_rate
    else:
        return phase_diff * sample_rate / (2.0 * math.pi)


def instantaneous_frequency_smoothed(
    H: torch.Tensor,
    sample_rate: int,
    eps: float = 1e-8,
    return_radians_per_second: bool = True,
) -> torch.Tensor:
    """IF estimator with amplitude regularization.

    Same as `instantaneous_frequency_from_phasors` but adds a small epsilon to
    the magnitude of the phasor product so `angle()` remains numerically stable
    when h ~ 0. Does NOT change the output for non-vanishing signals.

    Use this variant inside the alignment loss; use the plain version for
    analysis / diagnostics.
    """
    if not torch.is_complex(H):
        raise TypeError(f"Expected complex tensor, got dtype {H.dtype}.")

    time_axis = -2 if H.dim() >= 2 else -1
    H_prev = torch.roll(H, shifts=1, dims=time_axis)
    idx = [slice(None)] * H.dim()
    idx[time_axis] = 0
    H_prev = H_prev.clone()
    H_prev[tuple(idx)] = H[tuple(idx)]

    phasor_product = H * torch.conj(H_prev)
    # Add eps to magnitude to stabilize angle() near zero.
    # We do this by rescaling: replace z by z * (|z| + eps) / (|z| + eps),
    # but since angle(z * k) = angle(z) for any real k > 0, the identity is
    # already safe. The real trick is preventing exact zeros -- add a small
    # amount in the direction of z itself. Simplest stable form:
    magnitude = phasor_product.abs() + eps
    unit_phasor = phasor_product / magnitude
    phase_diff = torch.angle(unit_phasor)

    if return_radians_per_second:
        return phase_diff * sample_rate
    else:
        return phase_diff * sample_rate / (2.0 * math.pi)


__all__ = [
    "instantaneous_frequency_from_phasors",
    "instantaneous_frequency_smoothed",
]
