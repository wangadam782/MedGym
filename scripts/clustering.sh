#!/usr/bin/env bash
# Stage 0: Cluster patients by PINN-based behavioral similarity.
#
# Outputs: cohort_splits/$COHORT/cluster_map.csv
#          cohort_splits/$COHORT/patient_ids_cluster_order.txt
#
# Key variables (override via environment):
#   COHORT              cohort identifier, default cohort_1
#   PATIENT_IDS_CSV     training patient list CSV
#   IND_PINN_DIR        individual PINN root
#   PAIRWISE_DIR        output for pairwise distance matrices
#   GROUP_DIR           output for cluster group assignments
#   CLUSTER_OUT_DIR     output for final cluster map (cohort_splits/$COHORT)
#   GROUP_SIZE          patients per cluster (default 11 = 10 train + 1 test)
#   MAX_GROUPS          number of clusters (default 10)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python}"
COHORT="${COHORT:-cohort_1}"
CSV="${CSV:-data/mimic_pinn_v4_filtered.csv}"
PATIENT_IDS_CSV="${PATIENT_IDS_CSV:-checkpoints-cohort/${COHORT}/${COHORT}_training.csv}"
IND_PINN_DIR="${IND_PINN_DIR:-results/pinn/individual}"
PAIRWISE_DIR="${PAIRWISE_DIR:-results/clustering/${COHORT}/pairwise}"
GROUP_DIR="${GROUP_DIR:-results/clustering/${COHORT}/groups_k10}"
CLUSTER_OUT_DIR="${CLUSTER_OUT_DIR:-cohort_splits/${COHORT}}"
GROUP_SIZE="${GROUP_SIZE:-11}"
MAX_GROUPS="${MAX_GROUPS:-10}"
K_STEPS="${K_STEPS:-20}"
DEVICE="${DEVICE:-auto}"

echo "============================================================"
echo "  Stage 0: PINN clustering"
echo "  cohort          : $COHORT"
echo "  patient_ids_csv : $PATIENT_IDS_CSV"
echo "  pinn_dir        : $IND_PINN_DIR"
echo "  pairwise_dir    : $PAIRWISE_DIR"
echo "  group_dir       : $GROUP_DIR"
echo "  cluster_out_dir : $CLUSTER_OUT_DIR"
echo "  group_size      : $GROUP_SIZE  max_groups: $MAX_GROUPS"
echo "============================================================"

# Step 0a: Compute pairwise PINN distances
$PYTHON scripts/data_processing/compute_pinn_pairwise_dist.py \
    --csv             "$CSV" \
    --patient-dir     "$IND_PINN_DIR" \
    --out-dir         "$PAIRWISE_DIR" \
    --k-steps         "$K_STEPS" \
    --device          "$DEVICE" \
    --patient-ids-csv "$PATIENT_IDS_CSV"

# Step 0b: Find tight groups
$PYTHON scripts/data_processing/find_tight_groups.py \
    --in-dir    "$PAIRWISE_DIR" \
    --out-dir   "$GROUP_DIR" \
    --group-size "$GROUP_SIZE" \
    --max-groups "$MAX_GROUPS"

# Step 0c: Generate cluster map
mkdir -p "$CLUSTER_OUT_DIR"
$PYTHON scripts/data_processing/make_cluster_map.py \
    --input   "$GROUP_DIR/cluster_assignments.csv" \
    --out-dir "$CLUSTER_OUT_DIR"

echo
echo "Stage 0 complete."
echo "  cluster_map : $CLUSTER_OUT_DIR/cluster_map.csv"
