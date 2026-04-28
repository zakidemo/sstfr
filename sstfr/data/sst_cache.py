"""
In-memory loader for precomputed SST ridge caches.

Loads per-clip .npz files (produced by scripts/precompute_sst.py) into a
single dict keyed by clip basename, then serves per-batch tensors aligned
to the trainer's (B, L, K) convention.

Design choices:
  - Eager loading at init. Total cache for ESC-50 is ~25 MB (uncompressed
    in memory ~50 MB), trivial.
  - Targets are stored at hop rate (T_hop = L/hop_length = 500 for ESC-50)
    and upsampled by repetition at load time to match the alignment loss's
    sample-rate (L = 80000) convention.
  - Energy-based masking: frames whose ridge energy is below a per-clip
    relative threshold are masked out so the loss ignores them. The
    threshold is `mask_relative_floor * max_energy_for_that_ridge`.
  - Frequencies are converted from Hz to rad/s (the alignment loss's
    expected unit) at load time.

Usage:
    cache = SSTRidgeCache("data/cache/sst_ridges/esc50",
                          sample_rate=16000, hop_length=160)
    target, mask = cache.load_batch(filenames, device="cuda")
    # target: (B, L, K) float32 rad/s
    # mask:   (B, L, K) float32, 1 where ridge is meaningful, 0 elsewhere
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


class SSTRidgeCacheMissError(KeyError):
    """Raised when a clip's filename is not present in the cache."""


class SSTRidgeCache:
    """Eager in-memory loader for precomputed SST ridge targets.

    Args:
        cache_dir: path to the dataset cache root
            (e.g., "data/cache/sst_ridges/esc50"). Expected layout:
                <cache_dir>/meta.npz
                <cache_dir>/fold1/<basename>.npz
                ...
        sample_rate: audio sample rate in Hz. Must match the precomputation.
        hop_length: hop in samples used during precomputation.
        duration_seconds: audio clip length in seconds. Must match.
        mask_relative_floor: per-ridge energy threshold as a fraction of that
            ridge's peak energy in the clip. Frames with energy below this
            threshold are masked. Default 0.05 (5%).
        absolute_floor: an absolute energy floor in the same units as
            ridge_energies. Frames below this are also masked, regardless of
            relative threshold (handles all-silent clips). Default 1e-5.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        sample_rate: int = 16000,
        hop_length: int = 160,
        duration_seconds: float = 5.0,
        mask_relative_floor: float = 0.05,
        absolute_floor: float = 1e-5,
    ):
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.duration_samples = int(duration_seconds * sample_rate)
        self.t_hop = self.duration_samples // hop_length  # 500 for ESC-50
        self.mask_relative_floor = mask_relative_floor
        self.absolute_floor = absolute_floor

        # Load meta.npz (shared SST frequency axis + params)
        meta_path = self.cache_dir / "meta.npz"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Cache meta not found at {meta_path}. Did you run "
                f"`python scripts/precompute_sst.py --all-folds`?"
            )
        meta = np.load(meta_path, allow_pickle=True)
        self.ssq_freqs_hz = meta["ssq_freqs_hz"].astype(np.float32)
        cache_sr = int(meta["sample_rate"])
        cache_hop = int(meta["hop_length"])
        if cache_sr != sample_rate or cache_hop != hop_length:
            raise ValueError(
                f"Cache was built with sample_rate={cache_sr}, hop={cache_hop}, "
                f"but loader was constructed with sample_rate={sample_rate}, "
                f"hop={hop_length}. Recompute the cache or fix the loader args."
            )

        # Eager-load every clip cache into memory.
        # Storage: dict[basename] = (ridge_omegas, mask) at hop rate.
        # We pre-convert Hz -> rad/s and pre-build the mask here so per-batch
        # cost is just a dict lookup + repeat.
        self._entries: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        n_loaded = 0
        for fold_dir in sorted(self.cache_dir.glob("fold*")):
            for npz_path in fold_dir.glob("*.npz"):
                basename = npz_path.stem  # e.g., "1-100032-A-0"
                d = np.load(npz_path)

                ridge_freqs_hz = d["ridge_freqs_hz"]  # (K, T_hop) float32
                ridge_energies = d["ridge_energies"]  # (K, T_hop) float32

                if ridge_freqs_hz.shape[1] != self.t_hop:
                    raise ValueError(
                        f"{npz_path}: ridge length {ridge_freqs_hz.shape[1]} "
                        f"!= expected T_hop {self.t_hop}"
                    )

                # Convert Hz -> rad/s (the loss's unit)
                ridge_omegas = ridge_freqs_hz * (2.0 * math.pi)  # (K, T_hop)

                # Per-ridge relative + absolute energy mask
                # Shape (K, T_hop) bool -> float32
                per_ridge_max = ridge_energies.max(axis=1, keepdims=True)  # (K, 1)
                rel_threshold = self.mask_relative_floor * per_ridge_max
                effective_threshold = np.maximum(rel_threshold, self.absolute_floor)
                mask = (ridge_energies >= effective_threshold).astype(np.float32)

                self._entries[basename] = (ridge_omegas.astype(np.float32), mask)
                n_loaded += 1

        if n_loaded == 0:
            raise RuntimeError(
                f"No cache files found under {self.cache_dir}. "
                f"Did you run `python scripts/precompute_sst.py --all-folds`?"
            )

        # Sanity statistic
        self.n_clips = n_loaded
        self.n_ridges = next(iter(self._entries.values()))[0].shape[0]

    def __len__(self) -> int:
        return self.n_clips

    def __contains__(self, basename: str) -> bool:
        return basename in self._entries

    def load_batch(
        self,
        filenames: Iterable[str],
        device: torch.device | str = "cpu",
        upsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build target and mask tensors for a batch of clips.

        Args:
            filenames: iterable of B clip filenames (e.g., "1-100032-A-0.wav").
                Either with or without ".wav" extension; both are accepted.
            device: target device for the output tensors.
            upsample: if True, output is at sample rate (B, L, K) where
                L = duration * sample_rate. Each hop-rate value is repeated
                hop_length times. If False (default), output is at hop rate
                (B, T_hop, K). Hop rate is what the alignment loss should
                operate on for memory/speed reasons; sample-rate output is
                kept for backward compatibility and tests.

        Returns:
            target: (B, T, K) float32 ridge angular frequencies in rad/s.
                T = T_hop if upsample=False, else L.
            mask:   (B, T, K) float32 binary mask (1=valid frame, 0=masked).
        """
        filenames = list(filenames)
        B = len(filenames)
        if B == 0:
            raise ValueError("Empty filename list.")

        K = self.n_ridges
        T_hop = self.t_hop

        omegas_hop = np.empty((B, K, T_hop), dtype=np.float32)
        mask_hop = np.empty((B, K, T_hop), dtype=np.float32)

        for i, fname in enumerate(filenames):
            base = fname[:-4] if fname.endswith(".wav") else fname
            entry = self._entries.get(base)
            if entry is None:
                raise SSTRidgeCacheMissError(
                    f"No cache entry for clip '{fname}'. "
                    f"Cache size: {self.n_clips}. Run precomputation."
                )
            omegas_hop[i] = entry[0]
            mask_hop[i] = entry[1]

        if upsample:
            # Repeat each hop value hop_length times along the time axis.
            omegas_t = np.repeat(omegas_hop, self.hop_length, axis=2)
            mask_t = np.repeat(mask_hop, self.hop_length, axis=2)

            # Trim or zero-pad to exactly L (hop_length divides L cleanly for
            # ESC-50: 80000 / 160 = 500, but be defensive).
            L = self.duration_samples
            if omegas_t.shape[2] != L:
                if omegas_t.shape[2] > L:
                    omegas_t = omegas_t[:, :, :L]
                    mask_t = mask_t[:, :, :L]
                else:
                    pad = L - omegas_t.shape[2]
                    omegas_t = np.pad(omegas_t, ((0, 0), (0, 0), (0, pad)))
                    mask_t = np.pad(mask_t, ((0, 0), (0, 0), (0, pad)))
        else:
            omegas_t = omegas_hop
            mask_t = mask_hop

        # Reshape to (B, T, K) which is what the loss expects.
        target = torch.from_numpy(omegas_t.transpose(0, 2, 1).copy())
        mask = torch.from_numpy(mask_t.transpose(0, 2, 1).copy())

        if str(device) != "cpu":
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

        return target, mask


    def global_ridge_omegas(self) -> torch.Tensor:
        """Return a fixed (K,) tensor of mean ridge frequencies (rad/s) across
        the entire cached dataset, weighted by per-frame mask.

        Computed once at the first call and cached. Use this as the
        `ridge_omegas` argument to SynchrosqueezingAlignmentLoss so the
        ridge-to-channel assignment stays stable across training steps
        (a per-batch ridge_omegas would flicker, causing discontinuous
        loss values).

        Returns:
            (K,) float32 tensor on CPU. Caller may .to(device) it once.
        """
        if hasattr(self, "_global_omegas_cached"):
            return self._global_omegas_cached

        # Aggregate weighted sums across all clips
        K = self.n_ridges
        sum_weighted = np.zeros(K, dtype=np.float64)
        sum_weights = np.zeros(K, dtype=np.float64)
        for omegas, mask in self._entries.values():
            # omegas, mask: (K, T_hop)
            sum_weighted += (omegas * mask).sum(axis=1)
            sum_weights += mask.sum(axis=1)

        # Avoid division by zero on a fully-silent ridge across the dataset
        # (extremely unlikely but handle it gracefully)
        denom = np.where(sum_weights > 0, sum_weights, 1.0)
        global_omegas = (sum_weighted / denom).astype(np.float32)

        self._global_omegas_cached = torch.from_numpy(global_omegas)
        return self._global_omegas_cached


__all__ = ["SSTRidgeCache", "SSTRidgeCacheMissError"]
