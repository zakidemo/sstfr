"""
Experiment configuration dataclass.

Each experiment is fully described by a single `ExperimentConfig`. Configs are
saved alongside checkpoints and logged to TensorBoard so the run is fully
reproducible from the config alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one training run."""

    # --- Identity -------------------------------------------------------
    name: str = "logmel_esc50_fold1_seed0"
    frontend: Literal["logmel", "sstfr"] = "logmel"
    dataset: Literal["esc50"] = "esc50"
    fold: int = 1
    seed: int = 0

    # --- Data -----------------------------------------------------------
    data_root: str = "data/raw/ESC-50-master"
    sample_rate: int = 16000
    duration_seconds: float = 5.0
    num_workers: int = 4

    # --- Model ----------------------------------------------------------
    num_classes: int = 50
    num_channels: int = 128  # D (mel bins for logmel, SSM channels for sstfr)
    f_min: float = 40.0
    f_max: float | None = None  # None -> sample_rate/2
    # SSTFR-specific
    window_samples: int = 400
    decay_c: float = 4.0
    learn_alpha: bool = True
    learn_omega: bool = True
    learn_b: bool = True
    # Log-Mel specific
    n_fft: int = 1024
    hop_length: int = 160

    # --- Alignment loss (SSTFR only) -----------------------------------
    use_alignment_loss: bool = True
    lambda_ssa: float = 0.1
    sst_cache_dir: str = "data/cache/sst_ridges"
    # If use_alignment_loss=True but sst cache missing, we skip L_SSA gracefully
    detach_ssa_weights: bool = True
    cartesian_parameterization: bool = False  # ablation: a = alpha + i*omega instead of exp(.)

    # --- Optimization ---------------------------------------------------
    batch_size: int = 16
    num_epochs: int = 60
    lr_backbone: float = 1e-3
    lr_frontend_mult: float = 10.0  # SSTFR front-end gets 10x LR (paper)
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0

    # --- Runtime --------------------------------------------------------
    device: str = "cuda"
    amp: bool = False  # mixed precision (keep off for reproducibility unless needed)
    output_dir: str = "outputs"
    log_interval_batches: int = 10
    save_best_only: bool = True

    # --- Derived paths (filled in at runtime) --------------------------
    run_dir: str = ""  # <output_dir>/<name>

    def __post_init__(self) -> None:
        if not self.run_dir:
            self.run_dir = str(Path(self.output_dir) / self.name)
        if self.f_max is None:
            self.f_max = self.sample_rate / 2.0

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        # Only keep keys that are in the dataclass (allows old configs)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


__all__ = ["ExperimentConfig"]
