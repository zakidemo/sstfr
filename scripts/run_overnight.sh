#!/bin/bash
# Night 2B overnight batch: just the 7 SSTFR runs (Log-Mel done).
#
# Expected total: ~5 hours on RTX 4070 Laptop with K-quantized forward.

set -e

mkdir -p outputs
LOG="outputs/overnight_n2b.log"
echo "=============================================" | tee -a "$LOG"
echo "Night 2B overnight batch starting at $(date)" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"

CONFIGS=(
    # SSTFR fold 1 multi-seed (3 runs)
    "configs/sstfr_lam0_esc50_fold1_seed0.yaml"
    "configs/sstfr_lam0_esc50_fold1_seed1.yaml"
    "configs/sstfr_lam0_esc50_fold1_seed2.yaml"
    # SSTFR cross-fold (4 runs)
    "configs/sstfr_lam0_esc50_fold2_seed0.yaml"
    "configs/sstfr_lam0_esc50_fold3_seed0.yaml"
    "configs/sstfr_lam0_esc50_fold4_seed0.yaml"
    "configs/sstfr_lam0_esc50_fold5_seed0.yaml"
)

TOTAL=${#CONFIGS[@]}
COMPLETED=0
FAILED=0

for CONFIG in "${CONFIGS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    echo "" | tee -a "$LOG"
    echo "---------------------------------------------" | tee -a "$LOG"
    echo "[$(date)] Run $COMPLETED/$TOTAL: $CONFIG" | tee -a "$LOG"
    echo "---------------------------------------------" | tee -a "$LOG"

    if python scripts/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
        echo "[$(date)] OK: $CONFIG" | tee -a "$LOG"
    else
        FAILED=$((FAILED + 1))
        echo "[$(date)] FAILED: $CONFIG -- continuing" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"
echo "Batch complete at $(date)" | tee -a "$LOG"
echo "Completed: $COMPLETED/$TOTAL (failed: $FAILED)" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"

# Summary table
python << PYEOF | tee -a "$LOG"
import json
from pathlib import Path

print("\n=== FULL RESULTS TABLE ===\n")
print(f"{'name':<42}  {'val_acc':>8}  {'time_min':>8}")
print("-" * 64)

# Include all Log-Mel (already done) + all SSTFR (just ran)
names = [
    "logmel_esc50_fold1_seed0", "logmel_esc50_fold1_seed1", "logmel_esc50_fold1_seed2",
    "logmel_esc50_fold2_seed0", "logmel_esc50_fold3_seed0",
    "logmel_esc50_fold4_seed0", "logmel_esc50_fold5_seed0",
    "sstfr_lam0_esc50_fold1_seed0", "sstfr_lam0_esc50_fold1_seed1", "sstfr_lam0_esc50_fold1_seed2",
    "sstfr_lam0_esc50_fold2_seed0", "sstfr_lam0_esc50_fold3_seed0",
    "sstfr_lam0_esc50_fold4_seed0", "sstfr_lam0_esc50_fold5_seed0",
]
for name in names:
    summ = Path("outputs") / name / "summary.json"
    if summ.exists():
        d = json.loads(summ.read_text())
        print(f"{name:<42}  {d['best_val_acc']:>8.4f}  {d['elapsed_seconds']/60:>8.1f}")
    else:
        print(f"{name:<42}  MISSING")

# Aggregate
def mean_std(names):
    vals = []
    for n in names:
        f = Path("outputs") / n / "summary.json"
        if f.exists():
            vals.append(json.loads(f.read_text())['best_val_acc'])
    if not vals:
        return None
    m = sum(vals) / len(vals)
    s = (sum((v - m)**2 for v in vals) / len(vals)) ** 0.5
    return m, s, len(vals)

print()
print("=== AGGREGATES ===")
logmel_f1 = mean_std(["logmel_esc50_fold1_seed0", "logmel_esc50_fold1_seed1", "logmel_esc50_fold1_seed2"])
if logmel_f1:
    m, s, n = logmel_f1
    print(f"Log-Mel fold1 (3 seeds):    {m*100:.2f}% +/- {s*100:.2f}%  (n={n})")

logmel_5fold = mean_std([f"logmel_esc50_fold{i}_seed0" for i in range(1, 6)])
if logmel_5fold:
    m, s, n = logmel_5fold
    print(f"Log-Mel 5-fold (seed 0):    {m*100:.2f}% +/- {s*100:.2f}%  (n={n})")

sstfr_f1 = mean_std(["sstfr_lam0_esc50_fold1_seed0", "sstfr_lam0_esc50_fold1_seed1", "sstfr_lam0_esc50_fold1_seed2"])
if sstfr_f1:
    m, s, n = sstfr_f1
    print(f"SSTFR lam=0 fold1 (3 seeds): {m*100:.2f}% +/- {s*100:.2f}%  (n={n})")

sstfr_5fold = mean_std([f"sstfr_lam0_esc50_fold{i}_seed0" for i in range(1, 6)])
if sstfr_5fold:
    m, s, n = sstfr_5fold
    print(f"SSTFR lam=0 5-fold (seed 0): {m*100:.2f}% +/- {s*100:.2f}%  (n={n})")
PYEOF
