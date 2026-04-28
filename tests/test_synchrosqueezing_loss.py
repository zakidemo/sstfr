"""
Validation tests for SynchrosqueezingAlignmentLoss (Eq. 6 of the SSTFR paper).

We test each distinct claim of the loss design:
  1. Zero loss when SSM IF matches target IF exactly.
  2. Loss is monotonically decreasing as we interpolate toward the target.
  3. Gradient w.r.t. SSM channel frequency points toward the target frequency
     (i.e., the loss actually trains the right parameter).
  4. Amplitude weighting silences quiet regions: a loud well-aligned channel
     and a quiet misaligned channel should give low loss.
  5. Circular wraparound: frequencies near Nyquist do NOT produce exploding loss.
  6. No NaN/Inf on realistic audio-like inputs with silent regions.
  7. Ridge assignment is correct: nearest ridge is picked per channel.
"""

from __future__ import annotations

import math

import pytest
import torch

from sstfr.models.ssm_layer import SSTFRConfig, SSTFRLayer
from sstfr.losses.synchrosqueezing_loss import (
    SynchrosqueezingAlignmentLoss,
    assign_ridges_to_channels,
    circular_squared_difference,
)
from sstfr.losses.if_estimator import instantaneous_frequency_smoothed


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_complex_sinusoid(
    freq_hz: float, fs: int, L: int, amplitude: float = 1.0
) -> torch.Tensor:
    """A pure complex exponential at freq_hz, shape (L, 1), complex64."""
    omega = 2 * math.pi * freq_hz / fs
    t = torch.arange(L, dtype=torch.float32)
    return (amplitude * torch.exp(1j * omega * t)).unsqueeze(-1)  # (L, 1)


# --------------------------------------------------------------------------
# Test 1: Zero loss when perfect
# --------------------------------------------------------------------------

def test_zero_loss_when_ssm_matches_target():
    """If SSM IF == target IF everywhere, loss should be essentially zero."""
    fs = 16000
    B, L, D = 1, 500, 4
    freq_hz = 1000.0
    omega_rad_s = 2 * math.pi * freq_hz  # rad/s

    # Build H as a pure complex exponential (so its IF is exactly omega_rad_s)
    H = _make_complex_sinusoid(freq_hz, fs, L).unsqueeze(0)  # (1, L, 1)
    H = H.expand(B, L, D).contiguous()

    # Target: the same frequency for every (t, d)
    target = torch.full((B, L, D), omega_rad_s)
    channel_omegas = torch.full((D,), omega_rad_s / fs)  # irrelevant here

    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    loss = loss_fn(H, target, channel_omegas)
    print(f"\n  Loss when perfectly aligned: {loss.item():.6e}")
    # Expect near-zero (limited only by float32 roundoff in the IF estimator)
    assert loss.item() < 10.0, f"Loss too large for perfect alignment: {loss.item()}"


# --------------------------------------------------------------------------
# Test 2: Loss monotonically decreases as alignment improves
# --------------------------------------------------------------------------

def test_loss_decreases_as_target_approaches_ssm():
    """As target frequency moves toward SSM frequency, loss must decrease."""
    fs = 16000
    B, L, D = 1, 500, 1
    ssm_freq_hz = 2000.0
    target_freqs_hz = [500.0, 1000.0, 1500.0, 1900.0, 2000.0]

    H = _make_complex_sinusoid(ssm_freq_hz, fs, L).unsqueeze(0)
    H = H.expand(B, L, D).contiguous()
    channel_omegas = torch.tensor([2 * math.pi * ssm_freq_hz / fs])

    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    losses = []
    for f in target_freqs_hz:
        target = torch.full((B, L, D), 2 * math.pi * f)
        loss = loss_fn(H, target, channel_omegas).item()
        losses.append(loss)

    print(f"\n  Target freqs:  {target_freqs_hz}")
    print(f"  Losses:        {[f'{l:.2e}' for l in losses]}")

    # Must be monotonically decreasing
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i + 1], (
            f"Loss not decreasing: {losses[i]:.3e} -> {losses[i+1]:.3e}"
        )
    # Final value (perfect alignment) should be near zero
    assert losses[-1] < 10.0


# --------------------------------------------------------------------------
# Test 3: Gradient w.r.t. channel omega points toward target
# --------------------------------------------------------------------------

def test_gradient_w_r_t_channel_omega_points_to_target():
    """Using the full SSM layer: gradient of loss w.r.t. omega_d must move omega_d
    toward the target frequency (i.e., negative gradient direction moves it closer).
    """
    fs = 16000
    B, L = 1, 2000
    D = 1

    # Build an SSM with initial freq at 1500 Hz, target at 1000 Hz
    init_freq_hz = 1500.0
    target_freq_hz = 1000.0

    cfg = SSTFRConfig(
        num_channels=D,
        sample_rate=fs,
        f_min=init_freq_hz,
        f_max=init_freq_hz,
        window_samples=200,
    )
    layer = SSTFRLayer(cfg)

    # Input: sinusoid at target frequency so the SSM receives content there
    t = torch.arange(L, dtype=torch.float32) / fs
    x = torch.cos(2 * math.pi * target_freq_hz * t).unsqueeze(0)  # (1, L)

    H = layer(x)  # (1, L, 1)
    target = torch.full((B, L, D), 2 * math.pi * target_freq_hz)
    channel_omegas = layer.omega.detach().clone()

    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    loss = loss_fn(H, target, channel_omegas)
    loss.backward()

    # omega_init > omega_target, so d_loss/d_omega should be POSITIVE
    # (descending it = decreasing omega = moving toward target)
    grad_omega = layer.omega.grad.item()
    print(f"\n  init omega (rad/sample): {layer.omega.item():.4f}")
    print(f"  target omega (rad/sample): {2*math.pi*target_freq_hz/fs:.4f}")
    print(f"  gradient w.r.t. omega:    {grad_omega:.4e}")
    assert grad_omega > 0, (
        f"Gradient sign wrong: expected positive (to decrease omega toward target), "
        f"got {grad_omega:.3e}"
    )


# --------------------------------------------------------------------------
# Test 4: Amplitude weighting silences quiet regions
# --------------------------------------------------------------------------

def test_amplitude_weighting_silences_quiet_channels():
    """A channel with ~zero amplitude should contribute ~zero loss,
    regardless of its IF misalignment.
    """
    fs = 16000
    B, L = 1, 500

    # Two channels:
    #   d=0: loud signal at 1000 Hz, target 1000 Hz (perfectly aligned)
    #   d=1: ~zero signal, target very different (would be misaligned if weighted)
    H0 = _make_complex_sinusoid(1000.0, fs, L, amplitude=1.0)  # (L, 1)
    H1 = _make_complex_sinusoid(1000.0, fs, L, amplitude=1e-6)  # tiny
    H = torch.cat([H0, H1], dim=-1).unsqueeze(0)  # (1, L, 2)

    target = torch.zeros(B, L, 2)
    target[..., 0] = 2 * math.pi * 1000.0  # aligned
    target[..., 1] = 2 * math.pi * 5000.0  # wildly misaligned (but quiet)

    channel_omegas = torch.tensor([2 * math.pi * 1000.0 / fs, 2 * math.pi * 1000.0 / fs])

    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    loss = loss_fn(H, target, channel_omegas).item()

    print(f"\n  Loss with loud-aligned + quiet-misaligned: {loss:.3e}")
    # Loss should be dominated by the LOUD channel (which is aligned -> near zero)
    # Without amplitude weighting, the misaligned quiet channel would dominate.
    assert loss < 100.0, (
        f"Amplitude weighting failed: loss {loss:.3e} too high "
        f"(quiet misaligned channel shouldn't dominate)"
    )


# --------------------------------------------------------------------------
# Test 5: Circular wraparound does not explode the loss
# --------------------------------------------------------------------------

def test_wraparound_bounded_loss():
    """Two frequencies near the Nyquist boundary (one just below +pi*fs, one just
    above -pi*fs) are circularly CLOSE but linearly FAR. The loss must reflect
    the circular distance (small), not the linear one (huge).
    """
    fs = 16000
    nyquist = math.pi * fs  # rad/s

    # Frequencies separated by 0.01 rad/sample in circular metric
    # but ~2*pi*fs apart in linear
    omega_a = torch.tensor([nyquist - 0.005 * fs])
    omega_b = torch.tensor([-(nyquist - 0.005 * fs)])

    sq_err = circular_squared_difference(omega_a, omega_b, fs)
    linear_sq_err = (omega_a - omega_b) ** 2

    print(f"\n  Circular sq diff: {sq_err.item():.3e} (rad/s)^2")
    print(f"  Linear sq diff:   {linear_sq_err.item():.3e} (rad/s)^2")

    # Circular should be tiny (~(0.01 * fs)^2 = 2.56e4)
    # Linear should be ~(2*pi*fs)^2 = 1e10
    assert sq_err.item() < 1e6, f"Circular diff exploded: {sq_err.item()}"
    assert linear_sq_err.item() > 1e9, "Sanity: linear should be huge"


# --------------------------------------------------------------------------
# Test 6: Real SSM input with silent regions - no NaN/Inf
# --------------------------------------------------------------------------

def test_realistic_input_no_nan_no_inf():
    """Full pipeline: SSM on a signal with loud + silent regions. Loss must be
    finite and backward must produce finite gradients.
    """
    fs = 16000
    B, L = 2, 3000
    D = 16

    cfg = SSTFRConfig(num_channels=D, sample_rate=fs, window_samples=400)
    layer = SSTFRLayer(cfg)

    # Construct signal: sinusoid burst + silence + sinusoid burst
    x = torch.zeros(B, L)
    t = torch.arange(L, dtype=torch.float32) / fs
    burst1 = torch.cos(2 * math.pi * 1200.0 * t)
    burst2 = torch.cos(2 * math.pi * 2400.0 * t)
    # B=0: burst-silence-burst; B=1: steady sinusoid
    x[0, :1000] = burst1[:1000]
    x[0, 2000:] = burst2[2000:]
    x[1] = 0.3 * burst1

    H = layer(x)  # (B, L, D), complex

    # Fake target (per-channel pre-assigned)
    target = torch.randn(B, L, D) * 2 * math.pi * 2000.0
    channel_omegas = layer.omega.detach().clone()

    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    loss = loss_fn(H, target, channel_omegas)
    loss.backward()

    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    assert torch.isfinite(layer.omega.grad).all(), "omega gradient not finite"
    assert torch.isfinite(layer.alpha_raw.grad).all(), "alpha gradient not finite"
    assert torch.isfinite(layer.b_log_mag.grad).all(), "b_mag gradient not finite"
    assert torch.isfinite(layer.b_phase.grad).all(), "b_phase gradient not finite"
    print(f"\n  Realistic loss: {loss.item():.3e} -- all gradients finite")


# --------------------------------------------------------------------------
# Test 7: Ridge assignment picks the nearest ridge
# --------------------------------------------------------------------------

def test_ridge_assignment_nearest_neighbor():
    """Each channel must be assigned the ridge whose frequency is closest."""
    # Use well-separated values so there's no ambiguity
    channel_omegas = torch.tensor([0.1, 0.5, 1.2, 2.6])
    ridge_omegas = torch.tensor([0.2, 1.0, 2.5, 5.0])
    # Distances:
    #   ch 0 (0.1): to 0.2=0.1, to 1.0=0.9, to 2.5=2.4, to 5.0=4.9  -> ridge 0
    #   ch 1 (0.5): to 0.2=0.3, to 1.0=0.5, to 2.5=2.0, to 5.0=4.5  -> ridge 0
    #   ch 2 (1.2): to 0.2=1.0, to 1.0=0.2, to 2.5=1.3, to 5.0=3.8  -> ridge 1
    #   ch 3 (2.6): to 0.2=2.4, to 1.0=1.6, to 2.5=0.1, to 5.0=2.4  -> ridge 2
    assignment = assign_ridges_to_channels(channel_omegas, ridge_omegas)
    print(f"\n  channel freqs: {channel_omegas.tolist()}")
    print(f"  ridge freqs:   {ridge_omegas.tolist()}")
    print(f"  assignment:    {assignment.tolist()}")
    assert assignment[0].item() == 0
    assert assignment[1].item() == 0
    assert assignment[2].item() == 1
    assert assignment[3].item() == 2


# --------------------------------------------------------------------------
# Test 8: Ridge-indexed target form works end-to-end
# --------------------------------------------------------------------------

def test_ridge_indexed_target_form():
    """Calling convention B: target shape (B, L, K) with ridge_omegas."""
    fs = 16000
    B, L, D, K = 1, 200, 4, 3

    H = torch.randn(B, L, D, dtype=torch.complex64)
    target_ridges = torch.randn(B, L, K) * 2 * math.pi * 1000.0
    channel_omegas = torch.tensor([0.1, 0.5, 1.0, 2.0])
    ridge_omegas = torch.tensor([0.2, 1.1, 1.9])

    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    loss = loss_fn(H, target_ridges, channel_omegas, ridge_omegas=ridge_omegas)
    assert torch.isfinite(loss)
    print(f"\n  Ridge-indexed form loss: {loss.item():.3e}")


# ----------------------------------------------------------------------
# Tests for target_mask parameter (added when wiring in SSTRidgeCache)
# ----------------------------------------------------------------------


def _make_complex_H(B: int, L: int, D: int, magnitude: float = 1.0,
                    seed: int = 0) -> torch.Tensor:
    """Helper: build a complex tensor with a known magnitude profile."""
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(B, L, D, generator=g) * magnitude
    im = torch.randn(B, L, D, generator=g) * magnitude
    return torch.complex(re, im)


def test_mask_zero_eliminates_loss_contribution():
    """If target_mask is all zeros, the loss should be eps/eps ~ 0
    (denominator floored by eps so we don't divide by zero)."""
    fs = 16000
    B, L, D = 2, 100, 4
    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)

    H = _make_complex_H(B, L, D)
    target = torch.full((B, L, D), 2 * math.pi * 1000.0)  # 1 kHz target everywhere
    mask = torch.zeros(B, L, D)

    loss = loss_fn(H, target, channel_omegas=torch.zeros(D), target_mask=mask)
    # Numerator is 0, denominator is eps -> loss ~ 0
    assert loss.item() < 1e-3, f"expected near-zero loss, got {loss.item()}"


def test_mask_shape_mismatch_raises():
    fs = 16000
    B, L, D = 2, 100, 4
    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    H = _make_complex_H(B, L, D)
    target = torch.full((B, L, D), 2 * math.pi * 1000.0)
    bad_mask = torch.ones(B, L, D + 1)  # wrong D
    with pytest.raises(ValueError, match="target_mask"):
        loss_fn(H, target, channel_omegas=torch.zeros(D), target_mask=bad_mask)


def test_mask_subset_matches_unmasked_subset():
    """If we mask half the time axis, the loss equals what we'd get from
    running the unmasked loss on just the unmasked half."""
    fs = 16000
    B, L, D = 2, 100, 4
    torch.manual_seed(42)
    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)

    H = _make_complex_H(B, L, D, seed=42)
    target_omega = 2 * math.pi * 1000.0
    target = torch.full((B, L, D), target_omega)

    # Mask: keep first half, drop second half
    mask = torch.zeros(B, L, D)
    mask[:, : L // 2, :] = 1.0

    # Method 1: full tensors with mask
    loss_masked = loss_fn(H, target, channel_omegas=torch.zeros(D),
                          target_mask=mask)

    # Method 2: pre-trim H and target to the first half, no mask
    H_first = H[:, : L // 2, :]
    target_first = target[:, : L // 2, :]
    loss_unmasked = loss_fn(H_first, target_first, channel_omegas=torch.zeros(D))

    # They should agree to a tight tolerance.
    # NOTE: the masked version still has a t=0 row in the second half but
    # both versions have it in the first half -- so they should match.
    assert torch.isclose(loss_masked, loss_unmasked, rtol=1e-4, atol=1e-4), (
        f"masked: {loss_masked.item()}, equivalent unmasked: {loss_unmasked.item()}"
    )


def test_mask_works_with_ridge_indexed_target():
    """target_mask of shape (B, L, K) gets gathered alongside the target
    by the same ridge-to-channel assignment."""
    fs = 16000
    B, L, D, K = 2, 100, 4, 3
    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)

    torch.manual_seed(0)
    H = _make_complex_H(B, L, D)

    # K ridges at distinct frequencies
    ridge_omegas = torch.tensor([2 * math.pi * f for f in [500.0, 1500.0, 3000.0]])
    # Channel omegas closer to each of the ridges (deterministic assignment)
    channel_omegas = torch.tensor([2 * math.pi * f for f in [500.0, 500.0, 1500.0, 3000.0]])

    target = torch.zeros(B, L, K)
    target[:, :, 0] = 2 * math.pi * 500.0
    target[:, :, 1] = 2 * math.pi * 1500.0
    target[:, :, 2] = 2 * math.pi * 3000.0

    # Mask: full mask, ridge 0 active everywhere, ridge 1 inactive, ridge 2 active
    mask = torch.ones(B, L, K)
    mask[:, :, 1] = 0.0  # ridge 1 fully masked

    loss = loss_fn(H, target, channel_omegas, ridge_omegas, target_mask=mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_mask_does_not_break_gradients():
    """Gradients should still flow through SSM channel omegas with a mask."""
    fs = 16000
    B, L, D = 2, 100, 4
    torch.manual_seed(0)

    H = _make_complex_H(B, L, D, seed=0)
    target = torch.full((B, L, D), 2 * math.pi * 1500.0)
    mask = torch.zeros(B, L, D)
    mask[:, : L // 2, :] = 1.0

    channel_omegas = torch.full((D,), 2 * math.pi * 1000.0, requires_grad=True)
    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)

    # Note: in the (B, L, D) pre-assigned case, channel_omegas isn't actually
    # used by the loss (only by the ridge-indexed case). So we test gradients
    # via H instead.
    H.requires_grad_(True)
    loss = loss_fn(H, target, channel_omegas=channel_omegas, target_mask=mask)
    loss.backward()

    assert H.grad is not None
    assert torch.isfinite(H.grad).all()
    assert H.grad.abs().sum() > 0
