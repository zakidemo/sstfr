"""
ESC-50 dataset loader with official 5-fold cross-validation splits.

Dataset: https://github.com/karolpiczak/ESC-50
  - 2000 audio clips, 5 seconds each
  - 50 classes, 40 samples per class
  - Clips pre-assigned to folds 1-5 by the original authors
  - Official protocol: train on 4 folds, test on 1, repeat for all 5, report mean accuracy

Usage:
    from sstfr.data.esc50 import ESC50Dataset, ensure_esc50_downloaded

    root = ensure_esc50_downloaded("data/raw")
    train_ds = ESC50Dataset(root, fold=1, split="train", sample_rate=16000)
    test_ds = ESC50Dataset(root, fold=1, split="test", sample_rate=16000)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_ZIP_SIZE_MB = 600  # approximate


def ensure_esc50_downloaded(root_dir: str | Path) -> Path:
    """Ensure ESC-50 is downloaded and extracted. Returns the dataset root path.

    The ESC-50 repository extracts to `ESC-50-master/` containing:
        meta/esc50.csv    -- the metadata file (filename, fold, target, category, ...)
        audio/            -- the 2000 .wav files

    If the dataset is already present, this is a no-op. If not, downloads ~600MB
    zip and extracts.
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = root_dir / "ESC-50-master"

    if (dataset_root / "meta" / "esc50.csv").exists():
        audio_dir = dataset_root / "audio"
        n_wavs = len(list(audio_dir.glob("*.wav")))
        if n_wavs >= 2000:
            print(f"ESC-50 already present at {dataset_root} ({n_wavs} audio files).")
            return dataset_root
        print(f"WARNING: ESC-50 directory exists but has {n_wavs}/2000 audio files.")

    print(f"Downloading ESC-50 from {ESC50_URL} (~{ESC50_ZIP_SIZE_MB} MB)...")
    print("This takes 2-10 minutes depending on connection speed.")

    import requests

    response = requests.get(ESC50_URL, stream=True, timeout=60)
    response.raise_for_status()

    buffer = io.BytesIO()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    for chunk in response.iter_content(chunk_size=1 << 20):
        if chunk:
            buffer.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)",
                      end="", flush=True)
    print()
    buffer.seek(0)

    print("Extracting...")
    with zipfile.ZipFile(buffer) as zf:
        zf.extractall(root_dir)

    if not (dataset_root / "meta" / "esc50.csv").exists():
        raise RuntimeError(
            f"ESC-50 download extracted but meta/esc50.csv not found at {dataset_root}"
        )

    n_wavs = len(list((dataset_root / "audio").glob("*.wav")))
    print(f"Done. {n_wavs} audio files at {dataset_root}.")
    return dataset_root


class ESC50Dataset(Dataset):
    """ESC-50 audio classification dataset.

    Args:
        root: Path to the ESC-50-master directory.
        fold: Which fold (1-5) to use as the held-out test fold.
        split: "train" uses the other 4 folds; "test" uses the held-out fold.
        sample_rate: Target sample rate in Hz. Audio is resampled if needed.
            ESC-50 native is 44100 Hz; we downsample to 16000 for SSTFR.
        duration_seconds: Fixed clip duration. ESC-50 clips are 5 seconds;
            shorter clips are zero-padded, longer ones are center-cropped.
        normalize: If True (default), peak-normalize each waveform to [-1, 1].
        return_filename: If True, __getitem__ returns (wav, label, filename)
            instead of (wav, label). Used by the SST precomputation pipeline
            and the alignment-loss training path so each batch element can be
            mapped back to its precomputed cache file. Default False for
            backward compatibility.
    """

    NUM_CLASSES = 50

    def __init__(
        self,
        root: str | Path,
        fold: int,
        split: Literal["train", "test"],
        sample_rate: int = 16000,
        duration_seconds: float = 5.0,
        normalize: bool = True,
        return_filename: bool = False,
    ):
        if fold not in (1, 2, 3, 4, 5):
            raise ValueError(f"fold must be 1-5, got {fold}")
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.fold = fold
        self.split = split
        self.sample_rate = sample_rate
        self.duration_samples = int(duration_seconds * sample_rate)
        self.normalize = normalize
        self.return_filename = return_filename

        meta_path = self.root / "meta" / "esc50.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"ESC-50 metadata not found at {meta_path}. "
                f"Did you run ensure_esc50_downloaded()?"
            )
        meta = pd.read_csv(meta_path)

        if split == "test":
            meta = meta[meta["fold"] == fold]
        else:
            meta = meta[meta["fold"] != fold]

        self.meta = meta.reset_index(drop=True)

        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.meta)

    def _resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.sample_rate:
            return wav
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )
        return self._resamplers[orig_sr](wav)

    def _fix_length(self, wav: torch.Tensor) -> torch.Tensor:
        L = wav.shape[-1]
        target = self.duration_samples
        if L == target:
            return wav
        if L < target:
            pad = target - L
            return torch.nn.functional.pad(wav, (0, pad))
        start = (L - target) // 2
        return wav[..., start : start + target]

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        filepath = self.audio_dir / row["filename"]

        wav, sr = torchaudio.load(str(filepath))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = self._resample(wav, sr)
        wav = self._fix_length(wav)

        if self.normalize:
            peak = wav.abs().max()
            if peak > 1e-8:
                wav = wav / peak

        wav = wav.squeeze(0)
        label = int(row["target"])

        if self.return_filename:
            return wav, label, row["filename"]
        return wav, label


__all__ = ["ESC50Dataset", "ensure_esc50_downloaded"]
