#!/usr/bin/env bash
# sweep_pinn.sh — GPU-distributed loop over single-patient PINN training.
#
# Calls scripts/train_pinn.py once per patient.
#
# Supports:
#   - patient list from CSV / PATIENT_IDS / PATIENT_IDS_FILE
#   - machine-level round-robin sharding via MACHINE_ID / N_MACHINES
#   - GPU distribution via N_GPUS
#   - concurrent jobs per GPU via JOBS_PER_GPU
#   - resume via SKIP_DONE=1
#
# Output:
#   $SAVE_ROOT/patient_<pid>/
#     pinn.pt
#     medical_ode.pt
#     neural_ode.pt
#     scales.npy
#     _meta.json
#     run.log
#
# Examples
# --------
# All patients, 4 GPUs, 2 concurrent PINN jobs per GPU:
#   N_GPUS=4 JOBS_PER_GPU=2 SKIP_DONE=1 ./scripts/sweep_pinn.sh
#
# Restricted patients:
#   PATIENT_IDS="200325 201046 201101" N_GPUS=2 JOBS_PER_GPU=1 ./scripts/sweep_pinn.sh
#
# Patient list file:
#   PATIENT_IDS_FILE=results/paper/cohort/patient_ids.txt N_GPUS=4 JOBS_PER_GPU=2 ./scripts/sweep_pinn.sh
#
# Two-machine split:
#   MACHINE_ID=0 N_MACHINES=2 N_GPUS=4 JOBS_PER_GPU=2 ./scripts/sweep_pinn.sh
#   MACHINE_ID=1 N_MACHINES=2 N_GPUS=4 JOBS_PER_GPU=2 ./scripts/sweep_pinn.sh

set -u

# ─── Repo root (scripts/ → ..) ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ─── Python ──────────────────────────────────────────────────────────────────
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

# ─── Defaults ────────────────────────────────────────────────────────────────
CSV="${CSV:-data/mimic_pinn_v4_filtered.csv}"

# If data.config has PINN_IND_ROOT, use it. Otherwise fallback.
if [[ -z "${SAVE_ROOT:-}" ]]; then
  if PINN_IND_ROOT_VALUE="$("$PYTHON" - <<'PY' 2>/dev/null
from data.config import PINN_IND_ROOT
print(PINN_IND_ROOT)
PY
)"; then
    SAVE_ROOT="$PINN_IND_ROOT_VALUE"
  else
    SAVE_ROOT="results/pinn/individual"
  fi
fi

EPOCHS="${EPOCHS:-10000}"
PATIENCE="${PATIENCE:-800}"
LR_PINN="${LR_PINN:-1e-4}"
LR_ODE="${LR_ODE:-5e-5}"

N_GPUS="${N_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

MACHINE_ID="${MACHINE_ID:-0}"
N_MACHINES="${N_MACHINES:-1}"

SKIP_DONE="${SKIP_DONE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Sweep / multi-patient training uses the vectorized rollout loss by default.
# Set FAST_LOSS=0 to fall back to canonical loss.
FAST_LOSS="${FAST_LOSS:-1}"
ROLLOUT_N_STARTS="${ROLLOUT_N_STARTS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_pinn.py}"

if [[ ! -f "$CSV" ]]; then
  echo "[ERROR] CSV not found: $CSV" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] TRAIN_SCRIPT not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

N_SLOTS=$((N_GPUS * JOBS_PER_GPU))
if (( N_SLOTS <= 0 )); then
  echo "[ERROR] N_GPUS * JOBS_PER_GPU must be positive." >&2
  exit 1
fi

FAST_FLAG=""
if [[ "$FAST_LOSS" == "1" ]]; then
  FAST_FLAG="--fast_loss"
  if [[ -n "$ROLLOUT_N_STARTS" ]]; then
    FAST_FLAG="$FAST_FLAG --rollout_n_starts $ROLLOUT_N_STARTS"
  fi
fi

# ─── Discover patients ───────────────────────────────────────────────────────
if [[ -n "${PATIENT_IDS_FILE:-}" ]]; then
  if [[ ! -f "$PATIENT_IDS_FILE" ]]; then
    echo "[ERROR] PATIENT_IDS_FILE not found: $PATIENT_IDS_FILE" >&2
    exit 1
  fi
  mapfile -t ALL_PIDS < <(
    grep -E '^[0-9]+' "$PATIENT_IDS_FILE" \
      | tr -d '\r' \
      | awk '{print $1}' \
      | sort -n -u
  )
elif [[ -n "${PATIENT_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ALL_PIDS=( $PATIENT_IDS )
else
  mapfile -t ALL_PIDS < <(
    "$PYTHON" - <<PY
import pandas as pd
df = pd.read_csv("$CSV", usecols=["icu_id"])
ids = sorted(int(x) for x in df["icu_id"].dropna().unique().tolist())
print("\\n".join(str(i) for i in ids))
PY
  )
fi

if [[ ${#ALL_PIDS[@]} -eq 0 ]]; then
  echo "[ERROR] No patient ids resolved." >&2
  exit 1
fi

# ─── Build this-machine patient list ─────────────────────────────────────────
MY_PIDS=()
for i in "${!ALL_PIDS[@]}"; do
  if (( i % N_MACHINES == MACHINE_ID )); then
    MY_PIDS+=( "${ALL_PIDS[$i]}" )
  fi
done

is_done() {
  local pid="$1"
  [[ -f "$SAVE_ROOT/patient_${pid}/pinn.pt" ]]
}

run_one_patient() {
  local gpu="$1"
  local pid="$2"
  local out_dir="$SAVE_ROOT/patient_$pid"
  local log_path="$out_dir/run.log"

  if [[ "$SKIP_DONE" == "1" ]] && is_done "$pid"; then
    echo "[SKIP][GPU $gpu] patient $pid (pinn.pt exists)"
    return 0
  fi

  mkdir -p "$out_dir"

  echo
  echo "─── [GPU $gpu] patient $pid ──────────────────────────────────────────"
  echo "    out_dir: $out_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] would run train_pinn.py"
    return 0
  fi

  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$TRAIN_SCRIPT" \
      --patient_id "$pid" \
      --csv        "$CSV" \
      --save_dir   "$SAVE_ROOT" \
      --epochs     "$EPOCHS" \
      --patience   "$PATIENCE" \
      --lr_pinn    "$LR_PINN" \
      --lr_ode     "$LR_ODE" \
      $FAST_FLAG \
      $EXTRA_ARGS \
    2>&1 | tee -a "$log_path"

  local rc=${PIPESTATUS[0]}
  if (( rc != 0 )); then
    echo "[FAIL][GPU $gpu] patient $pid exited with rc=$rc" >&2
    return "$rc"
  fi

  echo "[DONE][GPU $gpu] patient $pid"
  return 0
}

run_slot() {
  local slot="$1"
  local gpu=$((slot / JOBS_PER_GPU))
  local n_run=0
  local n_fail=0

  for i in "${!MY_PIDS[@]}"; do
    if (( i % N_SLOTS == slot )); then
      local pid="${MY_PIDS[$i]}"
      if ! run_one_patient "$gpu" "$pid"; then
        n_fail=$((n_fail + 1))
      fi
      n_run=$((n_run + 1))
    fi
  done

  echo "[SLOT $slot] finished: run=$n_run fail=$n_fail"
  return "$n_fail"
}

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  sweep_pinn  GPU-distributed"
echo "  CSV           : $CSV"
echo "  TRAIN_SCRIPT  : $TRAIN_SCRIPT"
echo "  SAVE_ROOT     : $SAVE_ROOT"
echo "  EPOCHS        : $EPOCHS  PATIENCE=$PATIENCE"
echo "  LR_PINN       : $LR_PINN  LR_ODE=$LR_ODE"
echo "  FAST_LOSS     : $FAST_LOSS  ROLLOUT_N_STARTS='$ROLLOUT_N_STARTS'"
echo "  N_GPUS        : $N_GPUS"
echo "  JOBS_PER_GPU  : $JOBS_PER_GPU"
echo "  N_SLOTS       : $N_SLOTS"
echo "  MACHINE       : $MACHINE_ID / $N_MACHINES"
echo "  patients      : total=${#ALL_PIDS[@]}  this_machine=${#MY_PIDS[@]}"
echo "  SKIP_DONE     : $SKIP_DONE"
echo "  DRY_RUN       : $DRY_RUN"
echo "  PYTHON        : $PYTHON"
echo "  EXTRA_ARGS    : $EXTRA_ARGS"
echo "============================================================"

START_TS=$(date +%s)

pids=()
for slot in $(seq 0 $((N_SLOTS - 1))); do
  run_slot "$slot" &
  pids+=( "$!" )
done

N_FAIL_SLOTS=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    N_FAIL_SLOTS=$((N_FAIL_SLOTS + 1))
  fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo
echo "============================================================"
echo "  Done."
echo "  failed_slots : $N_FAIL_SLOTS"
printf "  elapsed      : %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
echo "  results      : $SAVE_ROOT"
echo "============================================================"

if (( N_FAIL_SLOTS != 0 )); then
  exit 1
fi
