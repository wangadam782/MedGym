#!/usr/bin/env bash
# Stage 1: Train PINN simulators (individual, cluster-pooled, population).
#
# Key variables:
#   COHORT              cohort identifier, default cohort_1
#   COHORT_DIR          checkpoints-cohort root, default checkpoints-cohort/$COHORT
#   CLUSTER_MAP         cluster assignment CSV
#   IND_PINN_DIR        output for individual PINNs  (legacy: results/pinn/individual)
#   POP_PINN_DIR        output for population PINN   (legacy: results/pinn/population)
#   SCOPE               which PINNs to train: all | individual | cluster | population
#   EPOCHS / PATIENCE   training budget (default 10000 / 800)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python}"
COHORT="${COHORT:-cohort_1}"
COHORT_DIR="${COHORT_DIR:-checkpoints-cohort/${COHORT}}"
CSV="${CSV:-data/mimic_pinn_v4_filtered.csv}"
PATIENT_IDS_CSV="${PATIENT_IDS_CSV:-checkpoints-cohort/${COHORT}/${COHORT}_training.csv}"
CLUSTER_MAP="${CLUSTER_MAP:-cohort_splits/${COHORT}/cluster_map.csv}"
IND_PINN_DIR="${IND_PINN_DIR:-${COHORT_DIR}/pinn/ind}"
CLU_PINN_DIR="${CLU_PINN_DIR:-${COHORT_DIR}/pinn/clu}"
POP_PINN_DIR="${POP_PINN_DIR:-${COHORT_DIR}/pinn/pop}"
SCOPE="${SCOPE:-all}"   # all | individual | cluster | population
EPOCHS="${EPOCHS:-10000}"
PATIENCE="${PATIENCE:-800}"
SKIP_DONE="${SKIP_DONE:-1}"

echo "============================================================"
echo "  Stage 1: PINN training"
echo "  cohort    : $COHORT"
echo "  scope     : $SCOPE"
echo "  epochs    : $EPOCHS / patience $PATIENCE"
echo "============================================================"

# ── Individual PINNs ─────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "individual" ]]; then
    echo
    echo "=== Individual PINNs ==="
    PYTHON="$PYTHON" \
    CSV="$CSV" \
    PATIENT_IDS_FILE="$PATIENT_IDS_CSV" \
    SAVE_ROOT="$IND_PINN_DIR" \
    SKIP_DONE="$SKIP_DONE" \
        bash scripts/sweep_pinn.sh
fi

# ── Population PINN ──────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "population" ]]; then
    echo
    echo "=== Population PINN ==="
    $PYTHON scripts/train_pinn_population.py \
        --csv            "$CSV" \
        --save_dir       "$POP_PINN_DIR" \
        --patient_ids_csv "$PATIENT_IDS_CSV" \
        --epochs         "$EPOCHS" \
        --patience       "$PATIENCE"
fi

# ── Cluster-pooled PINNs (one per cluster) ───────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "cluster" ]]; then
    echo
    echo "=== Cluster-pooled PINNs ==="
    if [[ ! -f "$CLUSTER_MAP" ]]; then
        echo "[ERROR] cluster_map not found: $CLUSTER_MAP" >&2
        echo "        Run clustering.sh first." >&2
        exit 1
    fi
    # Read unique cluster ids
    CLUSTER_IDS=$(tail -n +2 "$CLUSTER_MAP" | cut -d',' -f2 | sort -n | uniq)
    for CID in $CLUSTER_IDS; do
        SAVE_DIR="${CLU_PINN_DIR}/cluster_${CID}"
        if [[ "$SKIP_DONE" == "1" && -f "${SAVE_DIR}/pinn.pt" ]]; then
            echo "[SKIP] cluster ${CID}"
            continue
        fi
        # Collect training patient IDs for this cluster
        PIDS=$(awk -F',' -v cid="$CID" 'NR>1 && $2==cid && $3=="train" {print $1}' "$CLUSTER_MAP" | tr '\n' ' ')
        if [[ -z "$PIDS" ]]; then
            echo "[WARN] no training patients for cluster $CID, skipping"
            continue
        fi
        echo "  [cluster ${CID}] training patients: $PIDS"
        # shellcheck disable=SC2086
        $PYTHON scripts/train_pinn_population.py \
            --csv        "$CSV" \
            --save_dir   "$SAVE_DIR" \
            --patient_ids $PIDS \
            --epochs     "$EPOCHS" \
            --patience   "$PATIENCE"
    done
fi

echo
echo "Stage 1 complete."
