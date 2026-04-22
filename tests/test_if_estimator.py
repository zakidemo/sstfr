"""
Validation tests for the instantaneous frequency estimator.

Eq. (5) of the SSTFR paper:  omega_SSM(t,d) = f_s * arg(h_t * conj(h_{t-1}))

We verify:
  1. On a pure complex exponential, the estimator returns the exact frequency.
  2. On a linear chirp with known IF law omega(t) = omega_0 + beta*t, the
     estimator recovers the law to within discretization error.
  3. The estimator does not explode (returns finite values) in low-amplitude
     regions.
  4. The estimator is differentiable and gradients flow.
  5. The smoothed variant produces no NaNs when h contains exact zeros.
"""

from __future__ import annotations

import math

import pytest
import torch

from sstfr.losses.if_estimator import (
    instantaneous_frequency_from_phasors,
    instantaneous_frequency_smoothed,
)


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


# --------------------------------------------------------------------------
# Test 1: Pure complex exponential
# --------------------------------------------------------------------------

def test_pure_exponential_recovers_frequency_float64():
    """IF of exp(i*omega_0*t) must equal omega_0 exactly at float64 precision.

    This validates the MATHEMATICS of the estimator. At double precision, any
    nontrivial error would indicate a bug in the phasor-difference formula.
    """
    fs = 16000
    L = 4000
    f0_hz = 1234.5

    omega_0 = 2 * math.pi * f0_hz / fs
    t = torch.arange(L, dtype=torch.float64)
    H = torch.exp(1j * omega_0 * t).to(torch.complex128).unsqueeze(-1)

    omega_est = instantaneous_frequency_from_phasors(H, sample_rate=fs)
    f_est_hz = omega_est.squeeze(-1) / (2 * math.pi)

    mean_err = (f_est_hz[1:] - f0_hz).abs().mean().item()
    max_err = (f_est_hz[1:] - f0_hz).abs().max().item()
    print(f"\n  [float64] mean abs error: {mean_err:.3e} Hz")
    print(f"  [float64] max abs error:  {max_err:.3e} Hz")
    assert max_err < 1e-6, f"Max error {max_err} Hz indicates a math bug"


def test_pure_exponential_recovers_frequency_float32():
    """Same test in float32. Tolerance reflects single-precision roundoff.

    Phase grows linearly with t: at t=L=4000 and omega_0 ~ 0.48 rad/sample,
    the phase exceeds 1900 rad. float32 has ~7 decimal digits of precision,
    so accumulated argument error of ~1e-4 rad is expected. Converted to
    frequency, this yields ~1 Hz max error, which is negligible relative
    to any real audio classification task.
    """
    fs = 16000
    L = 4000
    f0_hz = 1234.5

    omega_0 = 2 * math.pi * f0_hz / fs
    t = torch.arange(L, dtype=torch.float32)
    H = torch.exp(1j * omega_0 * t).unsqueeze(-1)

    omega_est = instantaneous_frequency_from_phasors(H, sample_rate=fs)
    f_est_hz = omega_est.squeeze(-1) / (2 * math.pi)

    mean_err = (f_est_hz[1:] - f0_hz).abs().mean().item()
    max_err = (f_est_hz[1:] - f0_hz).abs().max().item()
    print(f"\n  [float32] mean abs error: {mean_err:.3e} Hz")
    print(f"  [float32] max abs error:  {max_err:.3e} Hz")
    # 1 Hz tolerance is far tighter than any physical requirement.
    assert max_err < 1.0, f"Max error {max_err} Hz too large"


# --------------------------------------------------------------------------
# Test 2: Linear chirp -- the textbook IF validation
# --------------------------------------------------------------------------

def test_linear_chirp_recovers_instantaneous_frequency():
    """For a chirp exp(i*(omega_0*t + beta*t^2/2)), the IF is omega_0 + beta*t."""
    fs = 16000
    L = 8000  # 0.5 s
    f0_hz = 500.0
    f1_hz = 3500.0  # chirp goes from 500 Hz to 3500 Hz over the duration

    # Chirp parameters
    t_seconds = torch.arange(L, dtype=torch.float64) / fs  # use float64 for precision
    duration = L / fs
    # Instantaneous frequency law: f(t) = f0 + (f1 - f0) * t / duration
    # Phase: phi(t) = 2*pi * integral(f(t')) = 2*pi * (f0*t + (f1-f0)*t^2 / (2*duration))
    phi = 2 * math.pi * (f0_hz * t_seconds + (f1_hz - f0_hz) * t_seconds**2 / (2 * duration))
    H = torch.exp(1j * phi).to(torch.complex64).unsqueeze(-1)  # (L, 1)

    omega_est = instantaneous_frequency_from_phasors(H, sample_rate=fs)
    f_est_hz = omega_est.squeeze(-1) / (2 * math.pi)

    # Expected IF
    f_true_hz = (f0_hz + (f1_hz - f0_hz) * t_seconds / duration).to(torch.float32)

    # Skip t=0 and t=L-1 (boundary effects)
    err = (f_est_hz[1:-1] - f_true_hz[1:-1]).abs()
    mean_err = err.mean().item()
    max_err = err.max().item()
    print(f"\n  mean abs error: {mean_err:.3f} Hz")
    print(f"  max abs error:  {max_err:.3f} Hz")

    # A well-implemented phasor IF on a clean chirp is essentially exact
    # (error is on the order of 1e-3 Hz). Give a generous 1 Hz tolerance.
    assert max_err < 1.0, f"Chirp IF error {max_err:.3f} Hz exceeds 1 Hz"


# --------------------------------------------------------------------------
# Test 3: Bounded output (no explosion)
# --------------------------------------------------------------------------

def test_output_is_bounded_by_nyquist():
    """|omega_est| must be <= pi*f_s (Nyquist bound)."""
    fs = 16000
    L = 2000
    H = torch.randn(L, 4, dtype=torch.complex64)
    omega_est = instantaneous_frequency_from_phasors(H, sample_rate=fs)
    assert omega_est.abs().max().item() <= math.pi * fs + 1e-3


# --------------------------------------------------------------------------
# Test 4: Differentiable
# --------------------------------------------------------------------------

def test_gradients_flow():
    """Gradient must flow through the estimator."""
    fs = 16000
    L = 200
    H = torch.randn(L, 4, dtype=torch.complex64, requires_grad=True)

    omega_est = instantaneous_frequency_from_phasors(H, sample_rate=fs)
    loss = (omega_est ** 2).mean()
    loss.backward()

    assert H.grad is not None
    assert torch.isfinite(H.grad).all(), "NaN/Inf in gradient"


# --------------------------------------------------------------------------
# Test 5: Smoothed variant handles exact zeros
# --------------------------------------------------------------------------

def test_smoothed_handles_zero_signal():
    """Smoothed IF must return finite values when H contains exact zeros."""
    fs = 16000
    L = 500
    H = torch.zeros(L, 4, dtype=torch.complex64)

    omega_est = instantaneous_frequency_smoothed(H, sample_rate=fs, eps=1e-8)
    assert torch.isfinite(omega_est).all(), "Smoothed IF produced NaN/Inf on zero input"


def test_smoothed_matches_plain_when_signal_nonzero():
    """On a non-vanishing signal, smoothed and plain must agree closely."""
    fs = 16000
    L = 1000
    f0_hz = 2000.0
    omega_0 = 2 * math.pi * f0_hz / fs
    t = torch.arange(L, dtype=torch.float32)
    H = torch.exp(1j * omega_0 * t).unsqueeze(-1)

    omega_plain = instantaneous_frequency_from_phasors(H, sample_rate=fs)
    omega_smooth = instantaneous_frequency_smoothed(H, sample_rate=fs, eps=1e-8)
    max_diff = (omega_plain - omega_smooth).abs().max().item()
    print(f"\n  max diff plain vs smoothed: {max_diff:.6e}")
    assert max_diff < 1e-3
