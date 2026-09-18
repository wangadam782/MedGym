#!/usr/bin/env bash
# Stage 2: Train online RL policies (individual, cluster-pooled, population).
#
# Key variables:
#   COHORT          cohort identifier, default cohort_1
#   COHORT_DIR      checkpoints-cohort root
#   SCOPE           ind | clu | pop | all  (default all)
#   ALGOS           space-separated algo names (default lagrangian_trpo)
#   K_VALUES        K values to train (default 20)
#   DT_MODES        fixdt vardt (default both)
#   SKIP_DONE       skip already-completed jobs (default 1)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python}"
COHORT="${COHORT:-cohort_1}"
COHORT_DIR="${COHORT_DIR:-checkpoints-cohort/${COHORT}}"
CLUSTER_MAP="${CLUSTER_MAP:-cohort_splits/${COHORT}/cluster_map.csv}"
SCOPE="${SCOPE:-all}"
ALGOS="${ALGOS:-lagrangian_trpo}"
K_VALUES="${K_VALUES:-20}"
DT_MODES="${DT_MODES:-fixdt vardt}"
SKIP_DONE="${SKIP_DONE:-1}"
N_GPUS="${N_GPUS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

echo "============================================================"
echo "  Stage 2: Online RL training"
echo "  cohort   : $COHORT"
echo "  scope    : $SCOPE"
echo "  algos    : $ALGOS"
echo "  K_VALUES : $K_VALUES"
echo "  dt_modes : $DT_MODES"
echo "============================================================"

# ── Individual policies ───────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "ind" ]]; then
    echo; echo "=== Individual policies ==="
    PYTHON="$PYTHON" \
    SCOPE=individual \
    PINN_DIR="${COHORT_DIR}/pinn/ind" \
    SAVE_ROOT="${COHORT_DIR}/online" \
    PATIENT_IDS_FILE="${CLUSTER_MAP}" \
    ALGOS="$ALGOS" K_VALUES="$K_VALUES" DT_MODES="$DT_MODES" \
    SKIP_DONE="$SKIP_DONE" N_GPUS="$N_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" \
        bash scripts/online/sweep_rl.sh
fi

# ── Population policy ─────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "pop" ]]; then
    echo; echo "=== Population policy ==="
    PYTHON="$PYTHON" \
    SCOPE=population \
    PINN_DIR="${COHORT_DIR}/pinn/pop" \
    SAVE_ROOT="${COHORT_DIR}/online" \
    ALGOS="$ALGOS" K_VALUES="$K_VALUES" DT_MODES="$DT_MODES" \
    SKIP_DONE="$SKIP_DONE" \
        bash scripts/online/sweep_rl.sh
fi

# ── Cluster-pooled policies ───────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "clu" ]]; then
    echo; echo "=== Cluster-pooled policies ==="
    PYTHON="$PYTHON" \
    SCOPE=cluster \
    PINN_DIR="${COHORT_DIR}/pinn/clu" \
    SAVE_ROOT="${COHORT_DIR}/online" \
    CLUSTER_TRAIN_SPLIT_CSV="$CLUSTER_MAP" \
    CLUSTER_INIT_PINN_DIR="${COHORT_DIR}/pinn/ind" \
    ALGOS="$ALGOS" K_VALUES="$K_VALUES" DT_MODES="$DT_MODES" \
    SKIP_DONE="$SKIP_DONE" N_GPUS="$N_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" \
        bash scripts/online/sweep_rl.sh
fi

echo
echo "Stage 2 complete."
