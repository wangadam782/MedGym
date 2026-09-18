#!/usr/bin/env bash
# Stage 3: Evaluate Ind / Cluster-pooled / Population online policies.
#
# Compares the three policy scopes on each cluster's test patient.
#
# Key variables:
#   COHORT      cohort identifier, default cohort_1
#   COHORT_DIR  checkpoints-cohort root
#   ALGO        algorithm to evaluate (default lagrangian_trpo)
#   K           K value (default 20)
#   N_EVAL      rollout episodes per patient (default 50)
#   SAVE_DIR    output directory

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python}"
COHORT="${COHORT:-cohort_1}"
COHORT_DIR="${COHORT_DIR:-checkpoints-cohort/${COHORT}}"
CLUSTER_MAP="${CLUSTER_MAP:-${COHORT_DIR}/${COHORT}_test.csv}"
ALGO="${ALGO:-lagrangian_trpo}"
K="${K:-20}"
N_EVAL="${N_EVAL:-50}"
SAVE_DIR="${SAVE_DIR:-results/online/eval/${COHORT}/${ALGO}/K${K}_n${N_EVAL}}"

echo "============================================================"
echo "  Stage 3: Online evaluation"
echo "  cohort   : $COHORT"
echo "  algo     : $ALGO   K=$K   n_eval=$N_EVAL"
echo "  save_dir : $SAVE_DIR"
echo "============================================================"

$PYTHON scripts/online/eval_ind_cluster_pop.py \
    --algo                "$ALGO" \
    --K                   "$K" \
    --n_eval              "$N_EVAL" \
    --pinn_dir            "${COHORT_DIR}/pinn/ind" \
    --ind_root            "${COHORT_DIR}/online/ind" \
    --pop_root            "${COHORT_DIR}/online/pop" \
    --cluster_root        "${COHORT_DIR}/online/clu" \
    --cluster_scales_root "${COHORT_DIR}/pinn/clu" \
    --glb_scales          "${COHORT_DIR}/pinn/pop/scales.npy" \
    --cluster_map         "$CLUSTER_MAP" \
    --save_dir            "$SAVE_DIR"

echo
echo "Stage 3 complete.  Results: $SAVE_DIR"
