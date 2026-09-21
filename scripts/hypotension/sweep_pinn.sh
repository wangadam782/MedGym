#!/usr/bin/env bash
# Train individual PINNs for patients LO..HI, then evaluate.
# Run from the medrl-tacos root.
#
#   bash scripts/hypotension/sweep_pinn.sh 0 9         # patients 0-9
#   bash scripts/hypotension/sweep_pinn.sh 0 9 6000    # custom epochs
set -euo pipefail
LO=${1:-0}; HI=${2:-59}; EPOCHS=${3:-4000}
for pid in $(seq "$LO" "$HI"); do
  echo "===== patient $pid ====="
  python scripts/hypotension/train_pinn.py --patient_id "$pid" --epochs "$EPOCHS" --quiet
done
python scripts/hypotension/eval_pinn.py --all --horizons 1 5
