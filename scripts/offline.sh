#!/usr/bin/env bash
# Stage 4: Cluster offline RL pipeline (5 steps).
#
#   Step 1  Collect Ind rollouts on each cluster test patient's PINN
#   Step 2  Copy cluster-pooled behavior datasets (rename fm_* → clu_*)
#   Step 3  Build population dataset from per-training-patient datasets
#   Step 4  Train DQN / CQL / GCQL for Ind / Clu / Pop in parallel
#   Step 5  Evaluate policies on test patients
#
# Key variables:
#   COHORT          cohort identifier, default cohort_1
#   COHORT_DIR      checkpoints-cohort root (provides ind PINNs + online ind policies)
#   OFFLINE_ROOT    output root for training; default results/offline_cluster/$COHORT
#   POLICY_ROOT     override for eval only; default $OFFLINE_ROOT/policies
#                   Set to checkpoints-cohort/$COHORT/offline to evaluate pre-trained checkpoints
#   ALGO            behavior policy algo (default lagrangian_trpo)
#   EPOCHS          offline training epochs (default 4000)
#   SKIP_DONE       skip already-completed jobs (default 1)
#
# Evaluate pre-trained checkpoints from HuggingFace download (skip steps 1-4):
#   COHORT=cohort_1 \
#   PINN_DIR=checkpoints-cohort/cohort_1/pinn/ind \
#   POLICY_ROOT=checkpoints-cohort/cohort_1/offline \
#   EVAL_ROOT=results/offline_eval/cohort_1 \
#   bash scripts/offline.sh --eval_only

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python}"
COHORT="${COHORT:-cohort_1}"
COHORT_DIR="${COHORT_DIR:-checkpoints-cohort/${COHORT}}"
CLUSTER_MAP="${CLUSTER_MAP:-${COHORT_DIR}/${COHORT}_test.csv}"
CLU_BY_PID="${CLU_BY_PID:-0}"
OFFLINE_ROOT="${OFFLINE_ROOT:-results/offline_cluster/${COHORT}}"
POLICY_ROOT="${POLICY_ROOT:-}"           # empty = use OFFLINE_ROOT/policies
EVAL_ROOT="${EVAL_ROOT:-}"               # empty = use OFFLINE_ROOT/eval
PINN_DIR="${PINN_DIR:-${COHORT_DIR}/pinn/ind}"
IND_ROOT="${IND_ROOT:-${COHORT_DIR}/online/ind}"
EVAL_ONLY="${1:-}"                       # pass --eval_only to skip steps 1-4
ALGO="${ALGO:-lagrangian_trpo}"
TAG="${TAG:-fixdt}"
K="${K:-20}"
N_EVAL_COLLECT="${N_EVAL_COLLECT:-50}"
N_EVAL_EVAL="${N_EVAL_EVAL:-50}"
N_CLUSTERS="${N_CLUSTERS:-10}"
EPOCHS="${EPOCHS:-4000}"
BATCH="${BATCH:-256}"
SKIP_DONE="${SKIP_DONE:-1}"

DSET="${OFFLINE_ROOT}/datasets"
PDIR="${OFFLINE_ROOT}/policies"

skip_flag() { [[ "$SKIP_DONE" == "1" ]] && echo "--skip_done" || echo "--no-skip_done"; }

echo "============================================================"
echo "  Stage 4: Cluster offline RL"
echo "  cohort        : $COHORT"
echo "  cluster_map   : $CLUSTER_MAP"
echo "  offline_root  : $OFFLINE_ROOT"
echo "  algo / tag    : $ALGO / $TAG   K=$K"
echo "  epochs        : $EPOCHS"
echo "============================================================"

if [[ "$EVAL_ONLY" == "--eval_only" ]]; then
    echo "[INFO] --eval_only: skipping steps 1-4 (data collection and training)"
else

# ── Step 1: Collect Ind rollouts for test patients ────────────────────────────
echo; echo "=== Step 1: Collect Ind rollouts ==="
$PYTHON scripts/offline/collect_cluster_test_data.py \
    --cluster_map  "$CLUSTER_MAP" \
    --out_dir      "$OFFLINE_ROOT" \
    --pinn_dir     "$PINN_DIR" \
    --ind_root     "$IND_ROOT" \
    --algo         "$ALGO" \
    --tag          "$TAG" \
    --K            "$K" \
    --n_eval       "$N_EVAL_COLLECT" \
    $(skip_flag)

# ── Steps 2/3: Prepare cluster + population datasets ─────────────────────────
echo; echo "=== Steps 2/3: Prepare datasets ==="
$PYTHON scripts/offline/prepare_offline_datasets.py \
    --out_dir    "$OFFLINE_ROOT" \
    --src_dir    "${DSET}/raw" \
    --tag        "$TAG" \
    --n_clusters "$N_CLUSTERS" \
    $(skip_flag)

# ── Step 4: Train DQN / CQL / GCQL × Ind / Clu / Pop ────────────────────────
echo; echo "=== Step 4: Train offline policies ==="
pids=()
for SCOPE in ind clu pop; do
    # Dataset tag prefix: clu uses "cluster" (not "clu") to match cluster{k}_fixdt naming
    TAG_PREFIX="${SCOPE}"; [[ "$SCOPE" == "clu" ]] && TAG_PREFIX="cluster"
    for METHOD_ALPHA in "dqn:0.0" "cql:5.0"; do
        METHOD="${METHOD_ALPHA%%:*}"; ALPHA="${METHOD_ALPHA##*:}"
        OUT="${PDIR}/${SCOPE}/${METHOD}"
        if [[ "$SKIP_DONE" == "1" ]] && find "$OUT/models" -name "actor_final.pt" -quit 2>/dev/null; then
            echo "[SKIP] ${SCOPE}/${METHOD}"; continue
        fi
        $PYTHON scripts/offline/train_cql.py \
            --dataset_dir "$DSET" --save_dir "$OUT" \
            --epochs "$EPOCHS" --batch_size "$BATCH" \
            --cql_alpha "$ALPHA" --tags "${TAG_PREFIX}" --no_mimic_csv &
        pids+=($!)
    done
    OUT="${PDIR}/${SCOPE}/gcql"
    if ! ( [[ "$SKIP_DONE" == "1" ]] && find "$OUT/models" -name "actor_final.pt" -quit 2>/dev/null ); then
        $PYTHON scripts/offline/train_gcql.py \
            --dataset_dir "$DSET" --save_dir "$OUT" \
            --epochs "$EPOCHS" --batch_size "$BATCH" \
            --tags "${TAG_PREFIX}" --no_mimic_csv &
        pids+=($!)
    fi
done
for p in "${pids[@]}"; do wait "$p"; done
echo "Step 4 complete."

fi  # end --eval_only guard

# ── Step 5: Evaluate ──────────────────────────────────────────────────────────
echo; echo "=== Step 5: Evaluate offline policies ==="
EVAL_ARGS=(
    --cluster_map  "$CLUSTER_MAP"
    --plan_a_root  "$OFFLINE_ROOT"
    --pinn_dir     "$PINN_DIR"
    --tag          "$TAG"
    --K            "$K"
    --n_eval       "$N_EVAL_EVAL"
    $(skip_flag)
)
[[ -n "$POLICY_ROOT"  ]] && EVAL_ARGS+=(--policy_root "$POLICY_ROOT")
[[ -n "$EVAL_ROOT"   ]] && EVAL_ARGS+=(--eval_root   "$EVAL_ROOT")
[[ "$CLU_BY_PID" == "1" ]] && EVAL_ARGS+=(--clu_by_pid)
$PYTHON scripts/offline/eval_cluster_offline.py "${EVAL_ARGS[@]}"

echo
echo "============================================================"
echo "  Stage 4 complete.  Results: $OFFLINE_ROOT"
echo "============================================================"
