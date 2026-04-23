"""
CLI entry point for training.

Usage:
    python scripts/train.py --config configs/logmel_esc50_fold1_seed0.yaml

Or import-and-call style:
    python -c "from sstfr.training.trainer import Trainer; from sstfr.training.config import ExperimentConfig; Trainer(ExperimentConfig.from_yaml('configs/foo.yaml')).fit()"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sstfr.training.config import ExperimentConfig
from sstfr.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override config fields: --override num_epochs=2 --override seed=1",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    # Apply overrides
    for kv in args.override:
        if "=" not in kv:
            raise ValueError(f"Bad override (need key=value): {kv}")
        key, val = kv.split("=", 1)
        if not hasattr(cfg, key):
            raise ValueError(f"Config has no field: {key}")
        # Cast to the field's current type
        current = getattr(cfg, key)
        if isinstance(current, bool):
            val = val.lower() in ("1", "true", "yes", "y")
        elif isinstance(current, int):
            val = int(val)
        elif isinstance(current, float):
            val = float(val)
        setattr(cfg, key, val)

    print(f"Training: {cfg.name}  (frontend={cfg.frontend}, fold={cfg.fold}, seed={cfg.seed})")
    trainer = Trainer(cfg)
    summary = trainer.fit()
    print(f"Done. Best val acc: {summary['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
