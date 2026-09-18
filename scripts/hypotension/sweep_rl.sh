#!/usr/bin/env bash
# Sweep all 42 LAG-TRPO training runs for the hypotension benchmark.
# Run from the medrl-tacos root.
#
# Scopes:
#   20 ind  (10 clusters x fixdt + vardt)
#   20 clu  (10 clusters x fixdt + vardt)
#    2 pop  (1           x fixdt + vardt)
#
# Usage:
#   bash scripts/hypotension/sweep_rl.sh
#   PARALLEL=4 bash scripts/hypotension/sweep_rl.sh
#   SCOPE=pop bash scripts/hypotension/sweep_rl.sh
#   SCOPE=ind CLUSTER_IDS="1 2" bash scripts/hypotension/sweep_rl.sh
set -euo pipefail

ALGO="${ALGO:-lagrangian_trpo}"
K="${K:-48}"
TOTAL_STEPS="${TOTAL_STEPS:-400000}"
HIDDEN="${HIDDEN:-128}"
DELTA="${DELTA:-0.02}"
LR_CRITIC="${LR_CRITIC:-3e-3}"
COST_LIMIT="${COST_LIMIT:-0.8}"
LR_LAGRANGE="${LR_LAGRANGE:-0.05}"
SCOPE="${SCOPE:-all}"
CLUSTER_IDS="${CLUSTER_IDS:-1 2 3 4 5 6 7 8 9 10}"
PARALLEL="${PARALLEL:-1}"

JOBS=()

_add_ind_clu() {
    local scope="$1"
    for cid in $CLUSTER_IDS; do
        for dt in fixdt vardt; do
            JOBS+=("--scope $scope --cluster_id $cid --dt_mode $dt")
        done
    done
}

_add_pop() {
    for dt in fixdt vardt; do
        JOBS+=("--scope pop --dt_mode $dt")
    done
}

case "$SCOPE" in
    ind) _add_ind_clu ind ;;
    clu) _add_ind_clu clu ;;
    pop) _add_pop ;;
    all) _add_ind_clu ind; _add_ind_clu clu; _add_pop ;;
    *) echo "Unknown SCOPE=$SCOPE (choices: ind clu pop all)"; exit 1 ;;
esac

TOTAL=${#JOBS[@]}
echo "============================================================"
echo "  hypotension sweep_rl.sh"
echo "  scope=$SCOPE  algo=$ALGO  K=$K"
echo "  total_steps=$TOTAL_STEPS  hidden=$HIDDEN  delta=$DELTA"
echo "  cost_limit=$COST_LIMIT  jobs=$TOTAL  parallel=$PARALLEL"
echo "============================================================"

run_job() {
    local args="$1"
    echo "--- START: $args ---"
    local steps="$TOTAL_STEPS"
    local dt_max_arg="--dt_max 6.0"
    if echo "$args" | grep -q "vardt"; then
        steps=$(( TOTAL_STEPS * 3 / 2 ))
    fi
    # shellcheck disable=SC2086
    python scripts/hypotension/train_rl.py \
        --algo        "$ALGO" \
        --K           "$K" \
        --total_steps "$steps" \
        --hidden      "$HIDDEN" \
        --delta       "$DELTA" \
        --lr_critic   "$LR_CRITIC" \
        --cost_limit  "$COST_LIMIT" \
        --lr_lagrange "$LR_LAGRANGE" \
        $dt_max_arg \
        $args
    echo "--- DONE:  $args ---"
}
export -f run_job
export ALGO K TOTAL_STEPS HIDDEN DELTA LR_CRITIC COST_LIMIT LR_LAGRANGE

printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_job "$@"' _ {}

echo ""
echo "All $TOTAL jobs completed."
