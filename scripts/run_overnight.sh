#!/bin/bash
# Overnight batch runner for SSTFR experiments.
# Runs configured experiments sequentially and logs stdout+stderr to outputs/overnight.log.
#
# Usage:
#   bash scripts/run_overnight.sh
# To run disconnected from the terminal (survives SSH disconnect):
#   nohup bash scripts/run_overnight.sh > outputs/overnight.log 2>&1 &
#   echo $! > outputs/overnight.pid

set -e  # exit on any unhandled error (we handle per-run errors below)

mkdir -p outputs
LOG="outputs/overnight.log"
echo "=============================================" | tee -a "$LOG"
echo "Overnight batch starting at $(date)" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"

# List of configs to run, in order (fast to slow)
CONFIGS=(
    "configs/logmel_esc50_fold1_seed1.yaml"
    "configs/logmel_esc50_fold1_seed2.yaml"
    "configs/sstfr_lam0_esc50_fold1_seed0.yaml"
    "configs/sstfr_lam0_esc50_fold1_seed1.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "" | tee -a "$LOG"
    echo "---------------------------------------------" | tee -a "$LOG"
    echo "[$(date)] Starting: $CONFIG" | tee -a "$LOG"
    echo "---------------------------------------------" | tee -a "$LOG"

    # Run in a subshell so a single run's failure doesn't kill the whole batch
    if python scripts/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
        echo "[$(date)] Completed: $CONFIG" | tee -a "$LOG"
    else
        echo "[$(date)] FAILED: $CONFIG -- continuing with next run" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"
echo "Overnight batch complete at $(date)" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"

# Summary table
echo "" | tee -a "$LOG"
echo "=== Results summary ===" | tee -a "$LOG"
for CONFIG in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG" .yaml)
    SUMMARY="outputs/${NAME}/summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "" | tee -a "$LOG"
        echo "--- $NAME ---" | tee -a "$LOG"
        cat "$SUMMARY" | tee -a "$LOG"
    else
        echo "" | tee -a "$LOG"
        echo "--- $NAME: NO SUMMARY (run may have failed) ---" | tee -a "$LOG"
    fi
done
