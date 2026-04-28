"""Unit tests for SSTRidgeCache."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from sstfr.data.sst_cache import SSTRidgeCache, SSTRidgeCacheMissError


CACHE_DIR = Path("data/cache/sst_ridges/esc50")
CACHE_AVAILABLE = (CACHE_DIR / "meta.npz").exists() and any(
    (CACHE_DIR / f"fold{i}").exists() for i in range(1, 6)
)


pytestmark = pytest.mark.skipif(
    not CACHE_AVAILABLE,
    reason="SST cache not precomputed; run scripts/precompute_sst.py first",
)


@pytest.fixture(scope="module")
def cache() -> SSTRidgeCache:
    return SSTRidgeCache(
        cache_dir=CACHE_DIR,
        sample_rate=16000,
        hop_length=160,
        duration_seconds=5.0,
    )


def test_loads_all_2000_clips(cache: SSTRidgeCache):
    assert len(cache) == 2000


def test_n_ridges_is_3(cache: SSTRidgeCache):
    assert cache.n_ridges == 3


def test_ssq_freqs_axis_loaded(cache: SSTRidgeCache):
    assert cache.ssq_freqs_hz.shape == (285,)
    assert cache.ssq_freqs_hz.min() > 0
    assert cache.ssq_freqs_hz.max() <= 8000.0 + 1.0  # Nyquist + tolerance


def test_contains_known_clip(cache: SSTRidgeCache):
    # Real ESC-50 clips that we cached
    assert "1-100038-A-14" in cache
    assert "1-100210-A-36" in cache


def test_does_not_contain_unknown_clip(cache: SSTRidgeCache):
    assert "nonexistent-clip-xyz" not in cache


def test_load_batch_shape(cache: SSTRidgeCache):
    filenames = ["1-100038-A-14.wav", "1-100210-A-36.wav"]
    target, mask = cache.load_batch(filenames, device="cpu")
    assert target.shape == (2, 500, 3)
    assert mask.shape == (2, 500, 3)
    assert target.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_load_batch_handles_extension_optional(cache: SSTRidgeCache):
    # With and without .wav should both work
    target_with, _ = cache.load_batch(["1-100038-A-14.wav"], device="cpu")
    target_without, _ = cache.load_batch(["1-100038-A-14"], device="cpu")
    assert torch.equal(target_with, target_without)


def test_target_frequencies_in_valid_range(cache: SSTRidgeCache):
    """All ridge omegas should be in [0, 2*pi*Nyquist] rad/s."""
    filenames = ["1-100038-A-14.wav", "1-100210-A-36.wav", "1-101296-A-19.wav"]
    target, _ = cache.load_batch(filenames, device="cpu")
    nyquist_omega = 2 * math.pi * 8000.0
    assert target.min() >= 0
    assert target.max() <= nyquist_omega + 1.0  # tolerance


def test_mask_is_binary(cache: SSTRidgeCache):
    """Mask values are exactly 0 or 1 (allowing float repr)."""
    filenames = ["1-100038-A-14.wav", "1-100210-A-36.wav"]
    _, mask = cache.load_batch(filenames, device="cpu")
    unique = torch.unique(mask)
    # Should be exactly {0, 1} (or just {1} or {0} for edge cases)
    for v in unique:
        assert v.item() in (0.0, 1.0), f"unexpected mask value: {v.item()}"


def test_mask_constant_within_hop(cache: SSTRidgeCache):
    """After hop-rate upsampling by repetition, the mask should be constant
    within each hop_length window."""
    filenames = ["1-100038-A-14.wav"]
    _, mask = cache.load_batch(filenames, device="cpu", upsample=True)
    # Reshape to (1, T_hop, hop_length, K) and check each hop window is uniform
    L = mask.shape[1]
    hop = 160
    K = mask.shape[2]
    T_hop = L // hop
    mask_reshaped = mask[0].view(T_hop, hop, K)  # (T_hop, hop, K)
    for t in range(T_hop):
        for k in range(K):
            window = mask_reshaped[t, :, k]
            assert window.min() == window.max(), (
                f"mask not constant in hop window t={t}, k={k}: "
                f"{window.unique().tolist()}"
            )


def test_target_constant_within_hop(cache: SSTRidgeCache):
    """Same constancy check for the target."""
    filenames = ["1-100038-A-14.wav"]
    target, _ = cache.load_batch(filenames, device="cpu", upsample=True)
    L = target.shape[1]
    hop = 160
    K = target.shape[2]
    T_hop = L // hop
    tgt_reshaped = target[0].view(T_hop, hop, K)
    for t in range(T_hop):
        for k in range(K):
            window = tgt_reshaped[t, :, k]
            assert window.min() == window.max()


def test_mask_density_varies_across_clips(cache: SSTRidgeCache):
    """Sanity check: not every clip has the same mask density."""
    filenames = [
        "1-100038-A-14.wav",   # birds, sparse
        "1-100210-A-36.wav",   # vacuum, dense
        "1-101296-A-19.wav",   # thunder
    ]
    _, mask = cache.load_batch(filenames, device="cpu")
    densities = [mask[i].mean().item() for i in range(len(filenames))]
    assert max(densities) - min(densities) > 0.1, (
        f"Mask densities too uniform: {densities}"
    )


def test_missing_clip_raises(cache: SSTRidgeCache):
    with pytest.raises(SSTRidgeCacheMissError):
        cache.load_batch(["this-clip-does-not-exist.wav"], device="cpu")


def test_empty_batch_raises(cache: SSTRidgeCache):
    with pytest.raises(ValueError):
        cache.load_batch([], device="cpu")


def test_init_rejects_mismatched_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        SSTRidgeCache(
            cache_dir=CACHE_DIR,
            sample_rate=8000,  # wrong! cache built at 16000
            hop_length=160,
        )


def test_init_rejects_mismatched_hop_length():
    with pytest.raises(ValueError, match="hop"):
        SSTRidgeCache(
            cache_dir=CACHE_DIR,
            sample_rate=16000,
            hop_length=80,  # wrong!
        )


def test_load_batch_upsample_true_returns_sample_rate(cache: SSTRidgeCache):
    """upsample=True should return (B, L, K) at sample rate, with each hop
    value repeated hop_length times."""
    filenames = ["1-100038-A-14.wav"]
    target_hop, mask_hop = cache.load_batch(filenames, device="cpu", upsample=False)
    target_full, mask_full = cache.load_batch(filenames, device="cpu", upsample=True)
    assert target_full.shape == (1, 80000, 3)
    assert mask_full.shape == (1, 80000, 3)
    # The first hop_length samples should equal the first hop value, repeated.
    hop = 160
    for k in range(3):
        assert torch.allclose(target_full[0, :hop, k],
                              target_hop[0, 0, k].expand(hop))
        assert torch.allclose(mask_full[0, :hop, k],
                              mask_hop[0, 0, k].expand(hop))


def test_init_rejects_missing_cache_dir():
    with pytest.raises(FileNotFoundError):
        SSTRidgeCache(
            cache_dir="/nonexistent/path",
            sample_rate=16000,
            hop_length=160,
        )
