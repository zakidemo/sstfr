"""
Numerical validation of Theorem 1 of the SSTFR paper.

Theorem 1 states: under the exponential parameterization a_d = exp(alpha_d + i*omega_d)
with alpha_d < 0, the SSM layer's hidden states are the output of an LTI filter with
impulse response g_d(tau) = b_d * a_d^tau for tau >= 0.

We verify this three ways:
  1. Impulse response matches the closed-form b_d * a_d^t.
  2. Linearity: L(ax + by) = a*L(x) + b*L(y).
  3. Frequency response: a pure sinusoid at f_0 Hz produces maximum amplitude in the
     channel whose center frequency is closest to f_0.

If any of these fail, Theorem 1 is wrong or the implementation has a bug.
"""

from __future__ import annotations

import math

import pytest
import torch

from sstfr.models.ssm_layer import SSTFRConfig, SSTFRLayer


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    """Ensure deterministic tests."""
    torch.manual_seed(0)


def _make_layer(D: int = 16, fs: int = 16000, window: int = 400) -> SSTFRLayer:
    cfg = SSTFRConfig(
        num_channels=D,
        sample_rate=fs,
        window_samples=window,
        decay_c=4.0,
    )
    return SSTFRLayer(cfg)


# --------------------------------------------------------------------------
# Test 1: Impulse response (the core of Theorem 1)
# --------------------------------------------------------------------------

def test_impulse_response_matches_closed_form():
    """Feeding delta[t] must yield h_t = b * a^t exactly (within float precision)."""
    D, L = 16, 500
    layer = _make_layer(D=D)

    # Impulse input: x = [1, 0, 0, ..., 0]
    x = torch.zeros(1, L)
    x[0, 0] = 1.0

    with torch.no_grad():
        H = layer(x).squeeze(0)  # (L, D)

    a = layer.a().detach()  # (D,)
    b = layer.b().detach()  # (D,)

    # Closed-form: h_t = b * a^t for t = 0, 1, ..., L-1
    t = torch.arange(L).unsqueeze(1)  # (L, 1)
    expected = b.unsqueeze(0) * (a.unsqueeze(0) ** t)  # (L, D)

    abs_err = (H - expected).abs().max().item()
    rel_err = abs_err / expected.abs().max().item()

    print(f"\n  max abs error: {abs_err:.3e}")
    print(f"  max rel error: {rel_err:.3e}")

    # Accumulated recurrence over 500 steps can have mild roundoff;
    # 1e-4 relative is comfortable below any physically meaningful threshold.
    assert abs_err < 1e-4, f"Impulse response mismatch: max abs error {abs_err:.3e}"
    assert rel_err < 1e-4, f"Relative error too large: {rel_err:.3e}"


def test_impulse_response_decays():
    """|h_t| must decay monotonically for an impulse (since |a| < 1)."""
    D, L = 8, 2000
    layer = _make_layer(D=D)

    x = torch.zeros(1, L)
    x[0, 0] = 1.0

    with torch.no_grad():
        H = layer(x).squeeze(0)  # (L, D)

    magnitudes = H.abs()  # (L, D)
    # Check monotonic decay for each channel
    for d in range(D):
        m = magnitudes[:, d]
        # Allow floating-point roundoff; strictly non-increasing within tolerance.
        diffs = m[1:] - m[:-1]
        assert (diffs <= 1e-7).all(), f"Channel {d}: impulse response not decaying"


# --------------------------------------------------------------------------
# Test 2: Linearity (consequence of LTI)
# --------------------------------------------------------------------------

def test_linearity():
    """L(a*x + b*y) = a*L(x) + b*L(y) for any scalars a, b."""
    layer = _make_layer(D=16)
    L = 800

    x = torch.randn(1, L)
    y = torch.randn(1, L)
    a_s, b_s = 2.3, -1.7

    with torch.no_grad():
        H_xy = layer(a_s * x + b_s * y)
        H_x = layer(x)
        H_y = layer(y)
        H_combo = a_s * H_x + b_s * H_y

    abs_err = (H_xy - H_combo).abs().max().item()
    print(f"\n  linearity max abs error: {abs_err:.3e}")
    assert abs_err < 1e-4, f"Linearity violated: {abs_err:.3e}"


# --------------------------------------------------------------------------
# Test 3: Frequency response (Gabor filter bank interpretation)
# --------------------------------------------------------------------------

def test_pure_sinusoid_activates_nearest_channel():
    """A sinusoid at f_0 Hz should maximally activate the channel tuned to f_0."""
    fs = 16000
    D = 32
    L = 4000  # 0.25 s at 16kHz -- enough for the filter to reach steady state

    layer = _make_layer(D=D, fs=fs)

    with torch.no_grad():
        target_freq_hz = 1000.0  # pick a test frequency
        t = torch.arange(L, dtype=torch.float32) / fs
        x = torch.cos(2 * math.pi * target_freq_hz * t).unsqueeze(0)  # (1, L)

        H = layer(x).squeeze(0)  # (L, D)

        # Use the second half of the response (past transient)
        steady_mag = H[L // 2 :].abs().mean(dim=0)  # (D,)
        best_channel = steady_mag.argmax().item()

        center_freqs = layer.center_frequencies_hz().detach()
        expected_channel = (center_freqs - target_freq_hz).abs().argmin().item()

        print(f"\n  Test freq: {target_freq_hz} Hz")
        print(f"  Best channel: {best_channel} ({center_freqs[best_channel]:.1f} Hz)")
        print(f"  Expected:     {expected_channel} ({center_freqs[expected_channel]:.1f} Hz)")

        assert best_channel == expected_channel, (
            f"Sinusoid at {target_freq_hz} Hz activated channel "
            f"{best_channel} ({center_freqs[best_channel]:.1f} Hz) instead of "
            f"channel {expected_channel} ({center_freqs[expected_channel]:.1f} Hz)"
        )


def test_frequency_response_shape():
    """The magnitude response vs. center frequency should peak at the input frequency."""
    fs = 16000
    D = 64
    L = 8000  # 0.5 s

    layer = _make_layer(D=D, fs=fs, window=200)  # wider bandwidth for smoother curve

    with torch.no_grad():
        target_freq_hz = 2000.0
        t = torch.arange(L, dtype=torch.float32) / fs
        x = torch.cos(2 * math.pi * target_freq_hz * t).unsqueeze(0)
        H = layer(x).squeeze(0)

        steady_mag = H[L // 2 :].abs().mean(dim=0)
        center_freqs = layer.center_frequencies_hz().detach()

        # Peak of the response should be near target_freq_hz
        peak_freq = center_freqs[steady_mag.argmax()].item()
        assert abs(peak_freq - target_freq_hz) / target_freq_hz < 0.15, (
            f"Response peak at {peak_freq:.1f} Hz, expected ~{target_freq_hz} Hz"
        )


# --------------------------------------------------------------------------
# Test 4: Alpha is always negative (safety check on the parameterization)
# --------------------------------------------------------------------------

def test_alpha_is_negative_after_optimization_step():
    """Even after a gradient step, alpha must remain < 0 thanks to -softplus."""
    layer = _make_layer(D=8)
    opt = torch.optim.Adam(layer.parameters(), lr=10.0)  # aggressive LR

    x = torch.randn(4, 500)
    for _ in range(5):
        opt.zero_grad()
        H = layer(x)
        loss = -H.abs().pow(2).mean()  # adversarial: try to blow up magnitudes
        loss.backward()
        opt.step()

    alpha = layer.alpha().detach()
    print(f"\n  alpha after adversarial training: {alpha}")
    assert (alpha < 0).all(), f"alpha became non-negative: {alpha}"
