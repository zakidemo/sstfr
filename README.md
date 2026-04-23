# SSTFR: State Space Time-Frequency Representations

Differentiable Gabor-like audio front-end based on complex-valued diagonal state space models, regularized by a synchrosqueezing alignment loss.

**Status:** Active development. Experiments in progress.

## Paper

> Anonymous Authors. *Differentiable State Space Time-Frequency Representations (SSTFR): Bridging Adaptive Filter Banks and Deep Audio Front-Ends.* Under review, 2026.

## Quick start

```bash
# Clone
git clone https://github.com/zakidemo/sstfr.git
cd sstfr

# Environment
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run a baseline
python -m sstfr.train --config configs/logmel_esc50.yaml
```

## License

Apache-2.0
