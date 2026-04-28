"""
Amplitude-weighted Synchrosqueezing Alignment Loss (L_SSA).

Implements Eq. (6) of the SSTFR paper:

    L_SSA = sum_{t,d} w_t^(d) * (omega_SSM(t,d) - omega_SST(t, k(d)))^2
            ---------------------------------------------------------------
                              sum_{t,d} w_t^(d) + eps

where:
  - h_t^(d) is the SSM hidden state (complex)
  - omega_SSM is the IF estimated from h via the phasor-difference formula
  - omega_SST is the precomputed synchrosqueezed target IF
  - k(d) is the ridge index assigned to SSM channel d
  - w_t^(d) = |h_t^(d)|^2 * target_mask_t^(d) * time_mask_t
  - eps prevents division by zero when the signal is silent

Design choices (note these in the paper revision):

  1. **Circular difference.** Eq. (6) as written uses (omega_SSM - omega_SST)^2,
     which blows up if the two frequencies are near opposite ends of
     [-pi*f_s, pi*f_s] (wraparound). We compute the squared distance in
     circular metric:
        d_circ(a, b)^2 = (f_s)^2 * (angle(exp(i*(a-b)/f_s)))^2
     This is equivalent to (a - b)^2 when the two values are close, and
     bounded by (pi*f_s)^2 otherwise. No gradient explosion.

  2. **Ridge assignment k(d).** At each training step, each SSM channel d is
     assigned the ridge index whose expected frequency is closest to the
     channel's current omega_d. The assignment is a discrete argmin (no
     gradient flows through the assignment itself), but gradient DOES flow
     through the alignment error once assigned. This matches the paper.

  3. **Amplitude weighting is non-differentiable in a useful sense.**
     The |h|^2 weights are *detached* from the computation graph of the IF
     misalignment. Otherwise the loss would encourage both channels with
     high amplitude to be well-aligned AND would drive channels to amplify
     themselves. We want only the former. This is standard in weighted
     losses (see e.g. focal loss).

  4. **Target mask.** Frames where the SST teacher has negligible energy
     produce ridge values that are essentially noise (the ridge-tracker's
     continuity prior fills in something, but it has no signal meaning).
     The optional `target_mask` parameter zeros out the loss contribution
     from those frames. The mask is produced by SSTRidgeCache based on
     per-ridge energy in the precomputed cache.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .if_estimator import instantaneous_frequency_smoothed


def circular_squared_difference(
    omega_a: torch.Tensor,
    omega_b: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Squared circular distance between two angular frequencies.

    Both omega_a and omega_b are in rad/s. Returns squared distance in (rad/s)^2.

    This is the same magnitude as (a - b)^2 when |a - b| < pi*f_s, but wraps
    gracefully at the Nyquist boundary. Equivalent to:
        (f_s * angle(exp(i * (omega_a - omega_b) / f_s)))^2
    """
    delta_per_sample = (omega_a - omega_b) / sample_rate  # rad/sample
    wrapped = torch.angle(torch.exp(1j * delta_per_sample.to(torch.complex64)))
    return (wrapped * sample_rate) ** 2


def assign_ridges_to_channels(
    channel_omegas: torch.Tensor,
    ridge_omegas: torch.Tensor,
) -> torch.Tensor:
    """Assign each SSM channel to the nearest ridge by center frequency.

    Args:
        channel_omegas: (D,) tensor of current SSM channel frequencies (rad/s).
        ridge_omegas: (K,) tensor of mean ridge frequencies (rad/s).

    Returns:
        assignment: (D,) long tensor of ridge indices in [0, K).
    """
    assert channel_omegas.dim() == 1, "channel_omegas must be 1D"
    assert ridge_omegas.dim() == 1, "ridge_omegas must be 1D"

    diff = (channel_omegas.unsqueeze(1) - ridge_omegas.unsqueeze(0))
    distance = diff.abs()
    return distance.argmin(dim=1)


class SynchrosqueezingAlignmentLoss(nn.Module):
    """Amplitude-weighted alignment loss (L_SSA).

    Usage (no mask, backward-compatible):
        loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=16000)
        loss = loss_fn(H, omega_target, channel_omegas, ridge_omegas)

    Usage (with target mask from SSTRidgeCache):
        loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=16000)
        target, target_mask = cache.load_batch(filenames, device=...)
        loss = loss_fn(H, target, channel_omegas, ridge_omegas,
                       target_mask=target_mask)

    Args:
        sample_rate: Sampling rate f_s in Hz.
        eps: Numerical stability constant for the amplitude denominator.
        detach_weights: If True (default), treat |h|^2 weights as constants
            for gradient flow (prevents the loss from being minimized trivially
            by shrinking |h|).
    """

    def __init__(
        self,
        sample_rate: int,
        eps: float = 1e-8,
        detach_weights: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.eps = eps
        self.detach_weights = detach_weights

    def forward(
        self,
        H: torch.Tensor,  # (B, L, D), complex
        omega_sst_target: torch.Tensor,  # (B, L, D) or (B, L, K), real, rad/s
        channel_omegas: torch.Tensor,  # (D,), real, rad/s
        ridge_omegas: torch.Tensor | None = None,  # (K,) if assignment needed
        target_mask: torch.Tensor | None = None,  # same shape as omega_sst_target
    ) -> torch.Tensor:
        """Compute the alignment loss.

        Two calling conventions for omega_sst_target (and optional target_mask):

        A) Pre-assigned: shape (B, L, D). For each channel d, target_sst[..., d]
           is already the target IF for that channel. Use when ridge assignment
           was done upstream. `ridge_omegas` is ignored.

        B) Ridge-indexed: shape (B, L, K). Requires `ridge_omegas` of shape (K,).
           We compute k(d) = assign_ridges_to_channels(channel_omegas, ridge_omegas)
           and gather target[..., k(d)] for each channel. `target_mask` (if
           provided) is gathered the same way.

        Args:
            H: complex tensor of shape (B, L, D), the SSM hidden states.
            omega_sst_target: real tensor of shape (B, L, D) or (B, L, K),
                target instantaneous frequencies in rad/s.
            channel_omegas: (D,) tensor of the SSM channels' center frequencies.
            ridge_omegas: (K,) tensor of mean ridge frequencies, required when
                target shape is (B, L, K).
            target_mask: optional real tensor of the same shape as
                omega_sst_target, with values in {0, 1}. Frames with mask=0
                are excluded from the loss (their amplitude weights are zeroed).

        Returns:
            scalar loss tensor.
        """
        if not torch.is_complex(H):
            raise TypeError(f"H must be complex, got {H.dtype}")
        if H.dim() != 3:
            raise ValueError(f"H must be (B, L, D), got {tuple(H.shape)}")

        B, L, D = H.shape

        # 1. Estimate IF of SSM hidden states
        omega_ssm = instantaneous_frequency_smoothed(
            H, sample_rate=self.sample_rate, eps=self.eps
        )  # (B, L, D), real rad/s

        # 2. Resolve the target IF (and optional mask) to shape (B, L, D)
        if omega_sst_target.shape == (B, L, D):
            target = omega_sst_target
            mask_for_loss = target_mask  # already (B, L, D) if provided
            if mask_for_loss is not None and mask_for_loss.shape != (B, L, D):
                raise ValueError(
                    f"target_mask shape {tuple(mask_for_loss.shape)} != "
                    f"target shape {(B, L, D)}"
                )
        else:
            if ridge_omegas is None:
                raise ValueError(
                    "ridge_omegas required when target shape is (B, L, K) "
                    "rather than (B, L, D)"
                )
            assignment = assign_ridges_to_channels(channel_omegas, ridge_omegas)  # (D,)
            idx = assignment.view(1, 1, D).expand(B, L, D)
            target = torch.gather(omega_sst_target, dim=-1, index=idx)
            if target_mask is not None:
                if target_mask.shape != omega_sst_target.shape:
                    raise ValueError(
                        f"target_mask shape {tuple(target_mask.shape)} != "
                        f"target shape {tuple(omega_sst_target.shape)}"
                    )
                mask_for_loss = torch.gather(target_mask, dim=-1, index=idx)
            else:
                mask_for_loss = None

        if target.shape != (B, L, D):
            raise ValueError(f"Target shape mismatch: {target.shape} vs {(B, L, D)}")

        # 3. Circular squared difference (prevents wraparound explosion)
        sq_err = circular_squared_difference(omega_ssm, target, self.sample_rate)
        # (B, L, D), in (rad/s)^2

        # 4. Amplitude weights (from the SSM side)
        weights = H.abs().pow(2)  # (B, L, D)
        if self.detach_weights:
            weights = weights.detach()

        # 5. Mask t=0: the phasor-difference IF estimator is undefined at t=0
        # (no previous sample), so its value there is meaningless. Including
        # it in the loss would punish the model for a mathematically
        # meaningless position.
        time_mask = torch.ones_like(weights)
        time_mask[:, 0, :] = 0.0
        weights = weights * time_mask

        # 6. Apply target mask (if any). Frames where the SST teacher has no
        # signal are excluded from the loss.
        if mask_for_loss is not None:
            weights = weights * mask_for_loss

        # 7. Amplitude-weighted mean
        numerator = (weights * sq_err).sum()
        denominator = weights.sum() + self.eps
        loss = numerator / denominator

        return loss


__all__ = [
    "SynchrosqueezingAlignmentLoss",
    "circular_squared_difference",
    "assign_ridges_to_channels",
]
