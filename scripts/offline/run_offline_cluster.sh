#!/usr/bin/env bash
# run_offline_cluster.sh
#
# Full cluster offline RL pipeline (5 steps):
#
#   Step 1  Collect Ind rollouts on each cluster test patient's PINN
#   Step 2  Copy existing cluster-pooled behavior datasets (rename fm_clu_* → cluster*)
#   Step 3  Build population dataset from per-training-patient datasets
#   Step 4  Train DQN / CQL / GCQL for Ind / Clu / Pop scopes in parallel
#   Step 5  Evaluate trained offline policies on test patients
#
# Requires:
#   - Individual PINN and online policies in results/pinn/individual/ and
#     results/online/individual/ (or override with --ind_root / --pinn_dir)
#   - Cluster behavior datasets (fm_clu_cluster*) in OFFLINE_ROOT/datasets/raw/
#   - cohort_splits/cohort_1/cluster_map.csv  (test patient assignments)
#
# Usage:
#   bash scripts/offline/run_offline_cluster.sh
#   COHORT=cohort_1 SKIP_DONE=1 bash scripts/offline/run_offline_cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
COHORT="${COHORT:-cohort_1}"

CLUSTER_MAP="${CLUSTER_MAP:-cohort_splits/${COHORT}/cluster_map.csv}"
PINN_DIR="${PINN_DIR:-results/pinn/individual}"
IND_ROOT="${IND_ROOT:-results/online/individual}"
OFFLINE_ROOT="${OFFLINE_ROOT:-results/offline_cluster/${COHORT}}"
ALGO="${ALGO:-lagrangian_trpo}"
TAG="${TAG:-fixdt}"
K="${K:-20}"
N_EVAL_COLLECT="${N_EVAL_COLLECT:-50}"
N_EVAL_EVAL="${N_EVAL_EVAL:-50}"
N_CLUSTERS="${N_CLUSTERS:-10}"
EPOCHS="${EPOCHS:-4000}"
BATCH="${BATCH:-256}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"

DSET="${OFFLINE_ROOT}/datasets"
PDIR="${OFFLINE_ROOT}/policies"

run_cmd() {
    echo
    printf '+'; printf ' %q' "$@"; echo
    [[ "$DRY_RUN" == "1" ]] && return 0
    "$@"
}

echo "============================================================"
echo "  Cluster Offline RL Pipeline"
echo "  cohort       : $COHORT"
echo "  cluster_map  : $CLUSTER_MAP"
echo "  pinn_dir     : $PINN_DIR"
echo "  offline_root : $OFFLINE_ROOT"
echo "  algo / tag   : $ALGO / $TAG"
echo "  epochs       : $EPOCHS"
echo "  dry_run      : $DRY_RUN"
echo "============================================================"

# ── Step 1: Collect Ind data for test patients ─────────────────────────────
echo
echo "=== Step 1: Collect Ind rollouts ==="
run_cmd "$PYTHON" scripts/offline/collect_cluster_test_data.py \
    --cluster_map  "$CLUSTER_MAP" \
    --out_dir      "$OFFLINE_ROOT" \
    --pinn_dir     "$PINN_DIR" \
    --ind_root     "$IND_ROOT" \
    --algo         "$ALGO" \
    --tag          "$TAG" \
    --K            "$K" \
    --n_eval       "$N_EVAL_COLLECT" \
    $( [[ "$SKIP_DONE" == "1" ]] && echo "--skip_done" || echo "--no-skip_done" )

# ── Step 2/3: Prepare Clu and Pop datasets ─────────────────────────────────
echo
echo "=== Steps 2/3: Prepare cluster + population datasets ==="
run_cmd "$PYTHON" scripts/offline/prepare_offline_datasets.py \
    --out_dir      "$OFFLINE_ROOT" \
    --src_dir      "${DSET}/raw" \
    --tag          "$TAG" \
    --n_clusters   "$N_CLUSTERS" \
    $( [[ "$SKIP_DONE" == "1" ]] && echo "--skip_done" || echo "--no-skip_done" )

# ── Step 4: Train offline policies ─────────────────────────────────────────
echo
echo "=== Step 4: Train offline policies (DQN / CQL / GCQL × Ind / Clu / Pop) ==="

for SCOPE in ind clu pop; do
    # clu scope datasets are named cluster{k}_fixdt (no "clu_" prefix)
    SCOPE_TAG="${SCOPE}"; [[ "$SCOPE" == "clu" ]] && SCOPE_TAG="cluster"
    for ALPHA in 0.0 5.0; do
        METHOD=$( [[ "$ALPHA" == "0.0" ]] && echo "dqn" || echo "cql" )
        OUT="${PDIR}/${SCOPE}/${METHOD}"
        if [[ "$SKIP_DONE" == "1" && -n "$(find "$OUT/models" -name 'actor_final.pt' 2>/dev/null)" ]]; then
            echo "[SKIP] $SCOPE/$METHOD"
            continue
        fi
        run_cmd "$PYTHON" scripts/offline/train_cql.py \
            --dataset_dir  "$DSET" \
            --save_dir     "$OUT" \
            --epochs       "$EPOCHS" \
            --batch_size   "$BATCH" \
            --cql_alpha    "$ALPHA" \
            --tags         "$SCOPE_TAG" \
            --no_mimic_csv &
    done

    OUT="${PDIR}/${SCOPE}/gcql"
    if [[ "$SKIP_DONE" == "1" && -n "$(find "$OUT/models" -name 'actor_final.pt' 2>/dev/null)" ]]; then
        echo "[SKIP] $SCOPE/gcql"
    else
        run_cmd "$PYTHON" scripts/offline/train_gcql.py \
            --dataset_dir  "$DSET" \
            --save_dir     "$OUT" \
            --epochs       "$EPOCHS" \
            --batch_size   "$BATCH" \
            --tags         "$SCOPE_TAG" \
            --no_mimic_csv &
    fi
done

wait
echo "Step 4 complete."

# ── Step 5: Evaluate ────────────────────────────────────────────────────────
echo
echo "=== Step 5: Evaluate offline policies on test patients ==="
run_cmd "$PYTHON" scripts/offline/eval_cluster_offline.py \
    --cluster_map  "$CLUSTER_MAP" \
    --plan_a_root  "$OFFLINE_ROOT" \
    --pinn_dir     "$PINN_DIR" \
    --tag          "$TAG" \
    --K            "$K" \
    --n_eval       "$N_EVAL_EVAL" \
    $( [[ "$SKIP_DONE" == "1" ]] && echo "--skip_done" || echo "--no-skip_done" )

echo
echo "============================================================"
echo "  Pipeline complete. Results: $OFFLINE_ROOT"
echo "============================================================"
