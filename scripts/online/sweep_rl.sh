#!/usr/bin/env bash
# sweep_rl.sh — GPU-scheduled launcher for scripts/online/train_rl.py.
#
# Trains online RL policies for all three scopes:
#   population  — one policy trained on the shared population PINN
#   individual  — one policy per test patient's individual PINN
#   cluster     — one pooled policy per cluster PINN
#
# Checkpoint layout:
#   ${SAVE_ROOT}/population/<algo>/K<K>/<fixdt|vardt>/policy.pt
#   ${SAVE_ROOT}/individual/patient_<id>/<algo>/K<K>/<fixdt|vardt>/policy.pt
#   ${SAVE_ROOT}/cluster_pooled/cluster_<id>/<algo>/K<K>/<fixdt|vardt>/policy.pt
#
# Examples:
#   SCOPE=population ./scripts/online/sweep_rl.sh
#   SCOPE=individual PATIENT_IDS_FILE=checkpoints-cohort/cohort_1/cohort_1_test.csv \
#     ./scripts/online/sweep_rl.sh
#   SCOPE=cluster CLUSTERS="1 2 3" ALGOS="sac lagrangian_trpo" K_VALUES="20" \
#     ./scripts/online/sweep_rl.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi

CONFIG="${CONFIG:-configs/online_rl/default.yaml}"

yaml_get() {
  local key="$1" default="$2"
  "$PYTHON" - "$CONFIG" "$key" "$default" <<'PY'
import sys
try:
    import yaml
except ImportError:
    print(sys.argv[3]); raise SystemExit(0)
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    print(default); raise SystemExit(0)
cur = cfg
for part in key.split("."):
    if not isinstance(cur, dict) or part not in cur:
        print(default); raise SystemExit(0)
    cur = cur[part]
if isinstance(cur, list):   print(" ".join(str(x) for x in cur))
elif isinstance(cur, bool): print("1" if cur else "0")
else:                       print(cur)
PY
}

# ── Shared settings ───────────────────────────────────────────────────────────
SCOPE="${SCOPE:-$(yaml_get sweep.scope population)}"
SAVE_ROOT="${SAVE_ROOT:-$(yaml_get paths.save_root results/online)}"
ALGOS="${ALGOS:-$(yaml_get sweep.algos 'sac ppo trpo lagrangian_ppo lagrangian_trpo')}"
K_VALUES="${K_VALUES:-$(yaml_get sweep.k_values '5 10 15 20')}"
DT_MODES="${DT_MODES:-$(yaml_get sweep.dt_modes 'fixdt vardt')}"
N_GPUS="${N_GPUS:-$(yaml_get sweep.n_gpus 4)}"
JOBS_PER_GPU="${JOBS_PER_GPU:-$(yaml_get sweep.jobs_per_gpu 2)}"
MACHINE_ID="${MACHINE_ID:-0}"
N_MACHINES="${N_MACHINES:-1}"
SKIP_DONE="${SKIP_DONE:-$(yaml_get sweep.skip_done 1)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

# ── Scope-specific settings ───────────────────────────────────────────────────
if [[ "$SCOPE" == "population" ]]; then
  PINN_DIR="${PINN_DIR:-$(yaml_get paths.population_pinn_dir results/pinn/population)}"

elif [[ "$SCOPE" == "individual" ]]; then
  PINN_DIR="${PINN_DIR:-$(yaml_get paths.individual_pinn_dir results/pinn/individual)}"

elif [[ "$SCOPE" == "cluster" ]]; then
  PINN_DIR="${PINN_DIR:-results/pinn/cluster_pooled}"
  CLUSTERS="${CLUSTERS:-1 2 3 4 5 6 7 8 9 10}"
  CLUSTER_TRAIN_SPLIT_CSV="${CLUSTER_TRAIN_SPLIT_CSV:-$(yaml_get paths.cluster_train_split_csv "")}"
  CLUSTER_INIT_PINN_DIR="${CLUSTER_INIT_PINN_DIR:-$(yaml_get paths.individual_pinn_dir results/pinn/individual)}"
  if [[ -z "$CLUSTER_TRAIN_SPLIT_CSV" ]]; then
    CLUSTER_TRAIN_SPLIT_CSV="cohort_splits/cohort_1/cluster_map.csv"
  fi
else
  echo "[ERROR] SCOPE must be 'population', 'individual', or 'cluster' (got: $SCOPE)" >&2
  exit 1
fi

if [[ ! -d "$PINN_DIR" ]]; then
  echo "[ERROR] PINN_DIR not found: $PINN_DIR" >&2; exit 1
fi

N_SLOTS=$((N_GPUS * JOBS_PER_GPU))
if (( N_SLOTS <= 0 )); then
  echo "[ERROR] N_GPUS * JOBS_PER_GPU must be positive." >&2; exit 1
fi

# ── Patient discovery (individual scope only) ─────────────────────────────────
PATIENT_LIST=()
if [[ "$SCOPE" == "individual" ]]; then
  if [[ -n "${PATIENT_IDS_FILE:-}" ]]; then
    [[ ! -f "$PATIENT_IDS_FILE" ]] && { echo "[ERROR] PATIENT_IDS_FILE not found: $PATIENT_IDS_FILE" >&2; exit 1; }
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && PATIENT_LIST+=( "$pid" )
    done < <(grep -E '^[0-9]+' "$PATIENT_IDS_FILE" | tr -d '\r' | awk '{print $1}' | sort -n -u)
  elif [[ -n "${PATIENT_IDS:-}" ]]; then
    PATIENT_LIST=( $PATIENT_IDS )
  else
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && PATIENT_LIST+=( "$pid" )
    done < <(find "$PINN_DIR" -mindepth 2 -maxdepth 2 -type f -name pinn.pt \
      | awk -F/ '{print $(NF-1)}' | sed -E 's/^patient_//' | sort -n -u)
  fi
  [[ ${#PATIENT_LIST[@]} -eq 0 ]] && { echo "[ERROR] No individual patients found under $PINN_DIR." >&2; exit 1; }
fi

# ── Policy path helper ────────────────────────────────────────────────────────
policy_path_for() {
  local scope="$1" pid="$2" cluster="$3" algo="$4" K="$5" dt="$6"
  case "$scope" in
    population) echo "$SAVE_ROOT/population/$algo/K$K/$dt/policy.pt" ;;
    individual) echo "$SAVE_ROOT/individual/patient_$pid/$algo/K$K/$dt/policy.pt" ;;
    cluster)    echo "$SAVE_ROOT/cluster_pooled/cluster_$cluster/$algo/K$K/$dt/policy.pt" ;;
  esac
}

# ── Single-job runner ─────────────────────────────────────────────────────────
run_one_job() {
  local gpu="$1" scope="$2" pid="$3" cluster="$4" algo="$5" K="$6" dt="$7"
  local policy_path save_dir log_path
  policy_path="$(policy_path_for "$scope" "$pid" "$cluster" "$algo" "$K" "$dt")"
  save_dir="$(dirname "$policy_path")"
  log_path="$save_dir/run.log"

  if [[ "$SKIP_DONE" == "1" && -f "$policy_path" ]]; then
    echo "[SKIP][GPU $gpu] $policy_path"; return 0
  fi
  mkdir -p "$save_dir"
  echo
  echo "─── [GPU $gpu] scope=$scope${pid:+ pid=$pid}${cluster:+ cluster=$cluster} algo=$algo K=$K dt=$dt"
  [[ "$DRY_RUN" == "1" ]] && { echo "[DRY_RUN]"; return 0; }

  if [[ "$scope" == "population" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/online/train_rl.py \
      --config "$CONFIG" --scope population \
      --algo "$algo" --K "$K" --dt_mode "$dt" \
      --pinn_dir "$PINN_DIR" --save_root "$SAVE_ROOT" --skip_done \
      $EXTRA_ARGS 2>&1 | tee -a "$log_path"
  elif [[ "$scope" == "individual" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/online/train_rl.py \
      --config "$CONFIG" --scope individual \
      --patient_id "$pid" --algo "$algo" --K "$K" --dt_mode "$dt" \
      --pinn_dir "$PINN_DIR" --save_root "$SAVE_ROOT" --skip_done \
      $EXTRA_ARGS 2>&1 | tee -a "$log_path"
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/online/train_rl.py \
      --config "$CONFIG" --scope cluster_pooled \
      --cluster_id "$cluster" --algo "$algo" --K "$K" --dt_mode "$dt" \
      --pinn_dir "$PINN_DIR" \
      --cluster_train_split_csv "$CLUSTER_TRAIN_SPLIT_CSV" \
      --cluster_init_pinn_dir "$CLUSTER_INIT_PINN_DIR" \
      --save_dir "$save_dir" --skip_done \
      $EXTRA_ARGS 2>&1 | tee -a "$log_path"
  fi

  local rc=${PIPESTATUS[0]}
  (( rc != 0 )) && { echo "[FAIL][GPU $gpu] rc=$rc" >&2; return "$rc"; }
  echo "[DONE][GPU $gpu] scope=$scope algo=$algo K=$K dt=$dt"
}

# ── Slot runner (parallel) ────────────────────────────────────────────────────
run_slot() {
  local slot="$1" gpu=$((slot / JOBS_PER_GPU)) job_idx=0 n_run=0 n_fail=0

  _try() {
    if (( job_idx % N_MACHINES != MACHINE_ID )); then job_idx=$((job_idx+1)); return; fi
    local local_idx=$((job_idx / N_MACHINES))
    if (( local_idx % N_SLOTS == slot )); then
      run_one_job "$gpu" "$@" || n_fail=$((n_fail+1)); n_run=$((n_run+1))
    fi
    job_idx=$((job_idx+1))
  }

  case "$SCOPE" in
    population)
      for algo in $ALGOS; do for K in $K_VALUES; do for dt in $DT_MODES; do
        _try population "" "" "$algo" "$K" "$dt"
      done; done; done ;;
    individual)
      for pid in "${PATIENT_LIST[@]}"; do for algo in $ALGOS; do for K in $K_VALUES; do for dt in $DT_MODES; do
        _try individual "$pid" "" "$algo" "$K" "$dt"
      done; done; done; done ;;
    cluster)
      for cluster in $CLUSTERS; do for algo in $ALGOS; do for K in $K_VALUES; do for dt in $DT_MODES; do
        _try cluster "" "$cluster" "$algo" "$K" "$dt"
      done; done; done; done ;;
  esac

  echo "[SLOT $slot] finished: run=$n_run fail=$n_fail"; return "$n_fail"
}

count_jobs() {
  local n_algos n_K n_dt n_units
  n_algos=$(wc -w <<< "$ALGOS"); n_K=$(wc -w <<< "$K_VALUES"); n_dt=$(wc -w <<< "$DT_MODES")
  case "$SCOPE" in
    population) n_units=1 ;;
    individual) n_units=${#PATIENT_LIST[@]} ;;
    cluster)    n_units=$(wc -w <<< "$CLUSTERS") ;;
  esac
  echo $((n_algos * n_K * n_dt * n_units))
}

echo "============================================================"
echo "  sweep_rl.sh"
echo "  SCOPE        : $SCOPE"
echo "  CONFIG       : $CONFIG"
echo "  PINN_DIR     : $PINN_DIR"
echo "  SAVE_ROOT    : $SAVE_ROOT"
[[ "$SCOPE" == "individual" ]] && echo "  PATIENTS     : ${#PATIENT_LIST[@]}"
[[ "$SCOPE" == "cluster"    ]] && echo "  CLUSTERS     : $CLUSTERS"
[[ "$SCOPE" == "cluster"    ]] && echo "  TRAIN_SPLIT  : $CLUSTER_TRAIN_SPLIT_CSV"
echo "  ALGOS        : $ALGOS"
echo "  K_VALUES     : $K_VALUES"
echo "  DT_MODES     : $DT_MODES"
echo "  N_GPUS       : $N_GPUS  JOBS_PER_GPU : $JOBS_PER_GPU  TOTAL : $(count_jobs)"
echo "  SKIP_DONE    : $SKIP_DONE  DRY_RUN : $DRY_RUN  PYTHON : $PYTHON"
echo "============================================================"

START_TS=$(date +%s)
pids=()
for slot in $(seq 0 $((N_SLOTS - 1))); do run_slot "$slot" & pids+=("$!"); done

N_FAIL=0
for p in "${pids[@]}"; do wait "$p" || N_FAIL=$((N_FAIL+1)); done

ELAPSED=$(( $(date +%s) - START_TS ))
echo; echo "============================================================"
echo "  sweep_rl.sh finished  failed_slots=$N_FAIL"
printf "  elapsed: %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
echo "  results: $SAVE_ROOT"
echo "============================================================"
(( N_FAIL != 0 )) && exit 1 || exit 0
