"""
Precompute synchrosqueezed-CWT ridges for ESC-50 clips.

For each clip:
  1. Load via ESC50Dataset (mirrors trainer's audio pipeline byte-for-byte).
  2. Compute ssq_cwt with default 'gmw' wavelet.
  3. Downsample Tx in time to hop rate, run extract_ridges (n_ridges=3).
  4. Save per-clip .npz with ridge frequencies and energies.

Output layout:
  <cache_dir>/
    meta.npz                  # ssq_freqs_hz, sst_params (one per dataset)
    fold1/
      <wav_basename>.npz       # one per clip
      ...
    fold2/
      ...

Per-clip .npz contains:
  ridge_freqs_hz: float32 (n_ridges, T_hop)
  ridge_energies: float32 (n_ridges, T_hop)
  ridge_idxs:     int32   (n_ridges, T_hop)  # for debugging only
  fold:           int
  target:         int
  sample_rate:    int
  hop_length:     int

Usage:
  python scripts/precompute_sst.py --fold 1
  python scripts/precompute_sst.py --all-folds
  python scripts/precompute_sst.py --fold 1 --overwrite

Notes:
  - The alignment loss expects targets at sample rate (B, L, K). The cache
    stores at hop rate (B, K, T_hop) for ~160x smaller files. The training
    loader upsamples (repeat) on the fly.
  - ridge_idxs are SST frequency-bin indices. ridge_freqs_hz are the
    corresponding center frequencies in Hz, looked up via ssq_freqs[idxs].
  - The meta.npz file stores ssq_freqs_hz once, since it depends only on
    sample rate and SST parameters (identical for every clip).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ssqueezepy import ssq_cwt, extract_ridges
from tqdm import tqdm

from sstfr.data.esc50 import ESC50Dataset


# ----------------------------------------------------------------------
# SST parameters (frozen here; if changed, all caches must be regenerated)
# ----------------------------------------------------------------------

SST_PARAMS = {
    "wavelet": "gmw",
    "scales": "log-piecewise",
    "nv": 32,
    "padtype": "reflect",
    "ridge_n": 3,
    "ridge_penalty": 2.0,
    "ridge_bw": 25,
}

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_LENGTH = 160
DEFAULT_DURATION_SECONDS = 5.0


def compute_ridges_for_clip(
    wav: np.ndarray,
    fs: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run SST + ridge extraction on a single waveform.

    Args:
        wav: float32 waveform of shape (L,).
        fs: sample rate.
        hop_length: temporal stride for downsampling Tx before ridge extraction.

    Returns:
        ridge_idxs:    int32   (n_ridges, T_hop)
        ridge_freqs:   float32 (n_ridges, T_hop) Hz
        ridge_energies:float32 (n_ridges, T_hop)
        ssq_freqs:     float32 (n_freqs,)  -- caller saves this once globally
    """
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    Tx, _Wx, ssq_freqs, scales = ssq_cwt(
        wav,
        fs=fs,
        wavelet=SST_PARAMS["wavelet"],
        scales=SST_PARAMS["scales"],
        nv=SST_PARAMS["nv"],
        padtype=SST_PARAMS["padtype"],
        astensor=False,
    )
    # Tx shape: (n_freqs, L)  complex64

    # Downsample in time before ridge extraction (160x faster, no quality loss)
    Tx_ds = Tx[:, ::hop_length]  # (n_freqs, T_hop)
    Tx_mag = np.abs(Tx_ds)        # float32

    # Forward-backward ridge tracking. Returns indices into the freq axis.
    # Returns shape (T_hop, n_ridges) per the docstring.
    ridge_idxs_t_first = extract_ridges(
        Tx_mag,
        scales=scales,
        penalty=SST_PARAMS["ridge_penalty"],
        n_ridges=SST_PARAMS["ridge_n"],
        bw=SST_PARAMS["ridge_bw"],
        transform="cwt",
        parallel=True,
    )
    # Transpose to (n_ridges, T_hop) for our cache convention
    ridge_idxs = ridge_idxs_t_first.T.astype(np.int32)  # (n_ridges, T_hop)

    # Convert indices -> Hz
    ridge_freqs = ssq_freqs[ridge_idxs].astype(np.float32)  # (n_ridges, T_hop)

    # Energy at each ridge point (for amplitude-aware target masking)
    n_ridges, T_hop = ridge_idxs.shape
    time_idx = np.arange(T_hop)[None, :].repeat(n_ridges, axis=0)  # (n_ridges, T_hop)
    ridge_energies = Tx_mag[ridge_idxs, time_idx].astype(np.float32)

    return ridge_idxs, ridge_freqs, ridge_energies, ssq_freqs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, default=None, help="Fold (1-5). If unset, requires --all-folds.")
    parser.add_argument("--all-folds", action="store_true", help="Process all 5 folds.")
    parser.add_argument("--data-root", type=str, default="data/raw/ESC-50-master")
    parser.add_argument("--cache-dir", type=str, default="data/cache/sst_ridges/esc50")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if cache exists.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N clips per fold (debug).")
    args = parser.parse_args()

    if args.fold is None and not args.all_folds:
        parser.error("Specify --fold N or --all-folds.")

    folds = list(range(1, 6)) if args.all_folds else [args.fold]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # We need ALL clips per fold (regardless of train/test split), because the
    # alignment loss runs on the train batches (clips from folds != test_fold).
    # Easiest: precompute ridges for every clip, indexed by source fold (1-5).
    # The trainer at test_fold=N will read caches from folds {1..5} \ {N}.

    # Track ssq_freqs across clips and confirm consistency.
    ssq_freqs_global: np.ndarray | None = None

    total_start = time.time()
    total_processed = 0
    total_skipped = 0

    for fold in folds:
        # ESC50Dataset's "test" subset gives us exactly the clips in `fold`.
        # We use this to enumerate every clip with its native fold label.
        ds = ESC50Dataset(
            root=args.data_root,
            fold=fold,
            split="test",  # gives clips IN this fold
            sample_rate=args.sample_rate,
            duration_seconds=args.duration_seconds,
            normalize=True,
            return_filename=True,
        )
        fold_dir = cache_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        n = len(ds) if args.limit is None else min(args.limit, len(ds))
        print(f"\n=== Fold {fold}: {n} clips ===")

        fold_start = time.time()
        pbar = tqdm(range(n), desc=f"fold {fold}")
        for i in pbar:
            wav, target, filename = ds[i]
            base = Path(filename).stem  # strip .wav
            out_path = fold_dir / f"{base}.npz"

            if out_path.exists() and not args.overwrite:
                total_skipped += 1
                continue

            wav_np = wav.numpy().astype(np.float32)
            try:
                ridge_idxs, ridge_freqs, ridge_energies, ssq_freqs = compute_ridges_for_clip(
                    wav_np, fs=args.sample_rate, hop_length=args.hop_length
                )
            except Exception as e:
                print(f"\n[error] {filename}: {e}")
                continue

            # Confirm ssq_freqs is identical across clips (it must be, since
            # we use the same SST params on the same-length signal).
            if ssq_freqs_global is None:
                ssq_freqs_global = ssq_freqs
            elif not np.array_equal(ssq_freqs_global, ssq_freqs):
                print(f"\n[warn] ssq_freqs changed at {filename}; this should not happen.")

            np.savez_compressed(
                out_path,
                ridge_idxs=ridge_idxs,
                ridge_freqs_hz=ridge_freqs,
                ridge_energies=ridge_energies,
                fold=np.int32(fold),
                target=np.int32(target),
                sample_rate=np.int32(args.sample_rate),
                hop_length=np.int32(args.hop_length),
            )
            total_processed += 1

            # Update tqdm postfix every 10 clips
            if i % 10 == 0:
                avg = (time.time() - fold_start) / max(1, i + 1)
                pbar.set_postfix({"avg_s/clip": f"{avg:.2f}"})
        pbar.close()

        elapsed = time.time() - fold_start
        print(f"Fold {fold}: {elapsed:.1f}s total, "
              f"{elapsed/max(1, n):.2f}s/clip avg")

    # Save shared meta (ssq_freqs + SST params)
    if ssq_freqs_global is not None:
        meta_path = cache_dir / "meta.npz"
        np.savez(
            meta_path,
            ssq_freqs_hz=ssq_freqs_global,
            sst_params_json=np.array(json.dumps(SST_PARAMS)),
            sample_rate=np.int32(args.sample_rate),
            hop_length=np.int32(args.hop_length),
            duration_seconds=np.float32(args.duration_seconds),
        )
        print(f"\nMeta saved to {meta_path} (n_freqs={len(ssq_freqs_global)})")

    total_elapsed = time.time() - total_start
    print(f"\nTotal: processed {total_processed}, skipped {total_skipped}, "
          f"elapsed {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
