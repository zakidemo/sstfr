"""
End-to-end integration test: ESC-50 loader -> SSTFR layer -> alignment loss.

This test requires ESC-50 to be present at data/raw/ESC-50-master. If not,
it is skipped (so CI on a fresh clone doesn't fail).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from sstfr.data.esc50 import ESC50Dataset
from sstfr.losses.synchrosqueezing_loss import SynchrosqueezingAlignmentLoss
from sstfr.models.ssm_layer import SSTFRConfig, SSTFRLayer


ESC50_ROOT = Path("data/raw/ESC-50-master")


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


@pytest.mark.skipif(
    not (ESC50_ROOT / "meta" / "esc50.csv").exists(),
    reason="ESC-50 dataset not downloaded. Run ensure_esc50_downloaded() first.",
)
def test_esc50_to_sstfr_to_alignment_loss():
    """One real batch flows end-to-end through the full forward pipeline."""
    fs = 16000
    D = 32  # smaller than production 128 to keep the test fast

    # --- Data ---
    ds = ESC50Dataset(str(ESC50_ROOT), fold=1, split="test", sample_rate=fs)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    wav, labels = next(iter(loader))  # (4, 80000)

    # For speed, take only the first 0.5 s (8000 samples)
    wav = wav[:, :8000]

    # --- Model ---
    cfg = SSTFRConfig(
        num_channels=D,
        sample_rate=fs,
        f_min=40.0,
        f_max=8000.0,
        window_samples=400,
    )
    layer = SSTFRLayer(cfg)
    H = layer(wav)  # (B, L, D) complex

    assert H.shape == (4, 8000, D)
    assert torch.is_complex(H)
    assert torch.isfinite(H.abs()).all()

    # --- Alignment loss with a synthetic target ---
    # Pretend each channel's target is its own initialized center frequency
    # (zero-error case, checking the loss plumbing works on real audio).
    target_freq_rad_s = layer.omega.detach() * fs  # (D,) rad/s
    target = target_freq_rad_s.view(1, 1, D).expand(4, 8000, D).contiguous()

    channel_omegas = layer.omega.detach()
    loss_fn = SynchrosqueezingAlignmentLoss(sample_rate=fs)
    loss = loss_fn(H, target, channel_omegas)

    print(f"\n  SSTFR output shape: {H.shape}")
    print(f"  Loss on aligned target (should be small): {loss.item():.3e}")

    assert torch.isfinite(loss)
    # The audio is not a pure sinusoid so IF fluctuates around center -- loss
    # is not zero, but should be finite and modest (< 1e10).
    assert loss.item() < 1e10, f"Loss unreasonably large: {loss.item():.3e}"


@pytest.mark.skipif(
    not (ESC50_ROOT / "meta" / "esc50.csv").exists(),
    reason="ESC-50 dataset not downloaded.",
)
def test_esc50_folds_are_disjoint():
    """Train fold and test fold must be disjoint and together cover all 2000 clips."""
    fs = 16000
    all_filenames: set[str] = set()
    overlaps_found = False

    for test_fold in (1, 2, 3, 4, 5):
        train_ds = ESC50Dataset(str(ESC50_ROOT), fold=test_fold, split="train", sample_rate=fs)
        test_ds = ESC50Dataset(str(ESC50_ROOT), fold=test_fold, split="test", sample_rate=fs)

        train_files = set(train_ds.meta["filename"])
        test_files = set(test_ds.meta["filename"])

        overlap = train_files & test_files
        if overlap:
            overlaps_found = True
            print(f"\n  Fold {test_fold}: {len(overlap)} overlapping files (BUG)")

        total = train_files | test_files
        assert len(train_files) == 1600, f"Fold {test_fold} train has {len(train_files)}"
        assert len(test_files) == 400, f"Fold {test_fold} test has {len(test_files)}"
        assert len(total) == 2000, f"Fold {test_fold} covers {len(total)} total"

        all_filenames.update(total)

    assert not overlaps_found, "Train/test overlap detected in some fold"
    assert len(all_filenames) == 2000
    print(f"\n  All 5 folds verified: disjoint train/test, 2000 total clips.")
