# Online RL Workflow

This document describes optional **online** RL policy training on PINN-backed simulators for the MedGym benchmark.

Released checkpoints live under:

```text
results/online/
results/pinn/
```

Retraining is **not** required to run the main benchmark if you use released artifacts. The commands below are for reproduction or extensions.

**Individual** training and the per-patient eval scripts require **`init_state_norm.npy`** next to each `pinn.pt` (PINN-normalised initial state). Wide-format MIMIC trajectory CSV is **not** read by `train_rl.py`, `eval_ind_cluster_pop.py`, or `eval_perpatient.py`. Optional YAML key `paths.csv` is only kept for `_meta.json` compatibility.

### Quick review

- **YAML** — Only `scripts/online/train_rl.py` (`--config`) and `scripts/online/sweep_rl.sh` (`CONFIG=...`) read `configs/online_rl/*.yaml`. PINN / eval shell scripts do not use these YAML files; set paths via env vars or CLI flags.
- **Smoke pipeline (cohort 1, SAC, K=20)** — Ensure PINNs exist under `results/pinn/`. Then train all three scopes, evaluate, and compare:

```bash
# 1) Individual SAC
SCOPE=individual PATIENT_IDS="219081 220949 228342" ALGOS=sac K_VALUES=20 ./scripts/online/sweep_rl.sh

# 2) Population SAC
SCOPE=population ALGOS=sac K_VALUES=20 ./scripts/online/sweep_rl.sh

# 3) Cluster-pooled SAC (K10 clusters)
SCOPE=cluster CLUSTERS="1 2 3 4 5 6 7 8 9 10" ALGOS=sac K_VALUES=20 ./scripts/online/sweep_rl.sh

# 4) Evaluate Ind / Cluster / Pop on cohort-7 test patients
python scripts/online/eval_ind_cluster_pop.py \
  --algo lagrangian_trpo --K 20 \
  --cluster_map cohort_splits/cohort_1/cluster_map.csv
```

Step 4 compares six checkpoints per patient (Ind/Clu/Pop × fixdt/vardt); missing weights are skipped with a warning.

---

## Expected inputs

### PINN simulator artifacts

Population PINN:

```text
results/pinn/population/
├── pinn.pt
└── scales.npy
```

Individual PINNs (one folder per patient):

```text
results/pinn/individual/patient_<pid>/
├── pinn.pt
├── scales.npy
└── init_state_norm.npy   # required for individual scope (no trajectory CSV)
```

Cluster-pooled PINNs (one folder per cluster):

```text
results/pinn/cluster_pooled/cluster_<id>/
├── pinn.pt
└── scales.npy
```

Paths above match `configs/online_rl/default.yaml`. Override with `PINN_DIR` env var or YAML `paths.*_pinn_dir`.

### Policy checkpoints (after training)

Under `SAVE_ROOT` from your YAML (examples use `results/online`):

```text
results/online/population/<algo>/K<K>/<fixdt|vardt>/
  policy.pt  actor.pt  _meta.json  train_log.csv

results/online/individual/patient_<pid>/<algo>/K<K>/<fixdt|vardt>/
  policy.pt  actor.pt  _meta.json  train_log.csv

results/online/cluster_pooled/cluster_<id>/<algo>/K<K>/<fixdt|vardt>/
  policy.pt  actor.pt  _meta.json  train_log.csv
```

`<algo>` is one of: `sac`, `ppo`, `trpo`, `lagrangian_ppo`, `lagrangian_trpo`.

---

## Sweep launcher environment variables

`scripts/online/sweep_rl.sh` accepts these overrides (defaults come from `CONFIG` YAML):

| Variable                 | Role                                                                              |
| ------------------------ | --------------------------------------------------------------------------------- |
| `CONFIG`                 | YAML path (default `configs/online_rl/default.yaml`).                             |
| `SCOPE`                  | `population`, `individual`, or `cluster`.                                         |
| `SAVE_ROOT`              | Overrides `paths.save_root`.                                                      |
| `PINN_DIR`               | Overrides PINN directory from YAML.                                               |
| `PATIENT_IDS_FILE`       | CSV/text file with patient IDs, one per line (`individual` scope).                |
| `PATIENT_IDS`            | Space-separated IDs if no file (`individual` scope).                              |
| `CLUSTERS`               | Space-separated cluster IDs, e.g. `1 2 3 4 5 6 7 8 9 10` (`cluster` scope).      |
| `CLUSTER_TRAIN_SPLIT_CSV`| CSV mapping patients to clusters and train/test split (`cluster` scope).          |
| `ALGOS`                  | Space-separated algorithm names (overrides `sweep.algos`).                        |
| `K_VALUES`               | Space-separated integers (overrides `sweep.k_values`).                            |
| `DT_MODES`               | e.g. `fixdt vardt` (overrides `sweep.dt_modes`).                                  |
| `N_GPUS`, `JOBS_PER_GPU` | Scheduling (overrides `sweep.n_gpus`, `sweep.jobs_per_gpu`).                      |
| `SKIP_DONE`              | `1` skips jobs whose `policy.pt` already exists (overrides `sweep.skip_done`).    |
| `EXTRA_ARGS`             | Extra flags forwarded to each `train_rl.py` invocation.                           |

Run from the **repository root** (the script `cd`s there).

---

## Step 1: Train population online policies

**In / Out:**

| Item            | Path                                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| Population PINN | `results/pinn/population/` → `pinn.pt`, `scales.npy`                                                             |
| Checkpoints     | `results/online/population/<algo>/K<K>/<fixdt|vardt>/policy.pt`, `actor.pt`, `_meta.json`, `train_log.csv`       |

Full sweep (all algos × K × dt from YAML):

```bash
SCOPE=population PINN_DIR=results/pinn/population SAVE_ROOT=results/online ./scripts/online/sweep_rl.sh
```

Cohort-7 single run (SAC, K=20, fixdt):

```bash
SCOPE=population ALGOS=sac K_VALUES=20 DT_MODES=fixdt N_GPUS=1 JOBS_PER_GPU=1 ./scripts/online/sweep_rl.sh
```

---

## Step 2: Train individual online policies

**Training cohort:** pass `PATIENT_IDS_FILE=checkpoints-cohort/cohort_1/cohort_1_training.csv` to train exactly the 110 cohort-7 patients.

**In / Out:**

| Item                     | Path                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Individual PINNs         | `results/pinn/individual/patient_<pid>/` with `pinn.pt`, `scales.npy`, `init_state_norm.npy`                                   |
| **Training cohort**      | `PATIENT_IDS_FILE=checkpoints-cohort/cohort_1/cohort_1_training.csv`                                                                |
| Checkpoints              | `results/online/individual/patient_<pid>/<algo>/K<K>/<fixdt|vardt>/policy.pt`, `actor.pt`, `_meta.json`, `train_log.csv`      |

Canonical command:

```bash
SCOPE=individual PINN_DIR=results/pinn/individual SAVE_ROOT=results/online \
  PATIENT_IDS_FILE=checkpoints-cohort/cohort_1/cohort_1_training.csv \
  ./scripts/online/sweep_rl.sh
```

Explicit smoke IDs (cohort-7 training patients):

```bash
SCOPE=individual PATIENT_IDS="200325 201046 201101" ALGOS=lagrangian_trpo K_VALUES=20 DT_MODES=fixdt \
  ./scripts/online/sweep_rl.sh
```

---

## Step 3: Train cluster-pooled online policies

One pooled policy per cluster PINN. Uses `SCOPE=cluster` in the unified `sweep_rl.sh`.

**In / Out:**

| Item               | Path                                                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| Cluster PINNs      | `results/pinn/cluster_pooled/cluster_<id>/` with `pinn.pt`, `scales.npy`                                                                |
| Train/test split   | `CLUSTER_TRAIN_SPLIT_CSV=cohort_splits/cohort_1/cluster_map.csv` (default)               |
| Checkpoints        | `results/online/cluster_pooled/cluster_<id>/<algo>/K<K>/<fixdt|vardt>/policy.pt`, `actor.pt`, `_meta.json`, `train_log.csv`            |

Full sweep:

```bash
SCOPE=cluster PINN_DIR=results/pinn/cluster_pooled SAVE_ROOT=results/online \
  CLUSTER_TRAIN_SPLIT_CSV=cohort_splits/cohort_1/cluster_map.csv \
  ./scripts/online/sweep_rl.sh
```

Single cluster smoke (cluster 1, SAC, K=20):

```bash
SCOPE=cluster CLUSTERS=1 ALGOS=sac K_VALUES=20 DT_MODES=fixdt N_GPUS=1 JOBS_PER_GPU=1 \
  ./scripts/online/sweep_rl.sh
```

---

## Step 4: Single-run training (`train_rl.py`)

One `(scope, algo, K, dt_mode)` without the sweep.

### Population

```bash
python scripts/online/train_rl.py --config configs/online_rl/default.yaml \
  --scope population --algo lagrangian_trpo --K 20 --dt_mode vardt \
  --pinn_dir results/pinn/population --save_root results/online
```

### Individual

```bash
python scripts/online/train_rl.py --config configs/online_rl/default.yaml \
  --scope individual --patient_id 219081 --algo lagrangian_trpo --K 20 --dt_mode fixdt \
  --pinn_dir results/pinn/individual --save_root results/online
```

### Cluster-pooled

```bash
python scripts/online/train_rl.py --config configs/online_rl/default.yaml \
  --scope cluster_pooled --cluster_id 1 --algo lagrangian_trpo --K 20 --dt_mode fixdt \
  --pinn_dir results/pinn/cluster_pooled \
  --cluster_train_split_csv cohort_splits/cohort_1/cluster_map.csv \
  --save_root results/online
```

---

## Step 5: Evaluation utilities

### Per-patient rollout sweep

`scripts/online/sweep_eval_perpatient.sh` drives `scripts/online/eval_perpatient.py` across GPUs. Outputs live under `${POLICY_ROOT}/eval/`.

```bash
PINN_DIR=results/pinn/individual POLICY_ROOT=results/online/individual \
  PATIENT_IDS_FILE=checkpoints-cohort/cohort_1/cohort_1_training.csv \
  ALGOS=sac K_VALUES=20 INIT_NOISE_STD=0 N_GPUS=1 JOBS_PER_GPU=1 \
  ./scripts/online/sweep_eval_perpatient.sh
```

### Ind / Cluster-pooled / Population comparison

Evaluates all three policy roles on cohort-7 test patients. Cluster-map CSV maps each test patient to its cluster.

```bash
python scripts/online/eval_ind_cluster_pop.py \
  --algo lagrangian_trpo --K 20 \
  --pinn_dir          results/pinn/individual \
  --ind_root          results/online/individual \
  --pop_root          results/online/population \
  --cluster_root      results/online/cluster_pooled \
  --cluster_scales_root results/pinn/cluster_pooled \
  --glb_scales        results/pinn/population/scales.npy \
  --cluster_map       cohort_splits/cohort_1/cluster_map.csv \
  --n_eval 50 --save_dir results/online/eval_cohort7
```

Smoke pipeline (train all 3 scopes → evaluate):

```bash
SCOPE=population  ALGOS=sac K_VALUES=20 ./scripts/online/sweep_rl.sh
SCOPE=individual  PATIENT_IDS="219081 220949" ALGOS=sac K_VALUES=20 ./scripts/online/sweep_rl.sh
SCOPE=cluster     CLUSTERS="1 2" ALGOS=sac K_VALUES=20 ./scripts/online/sweep_rl.sh

python scripts/online/eval_ind_cluster_pop.py --algo sac --K 20 \
  --cluster_map cohort_splits/cohort_1/cluster_map.csv \
  --patient_ids 219081 220949 --n_eval 5 \
  --save_dir test/online/eval_cohort7
```

---

## Main benchmark evaluation

End-to-end benchmark evaluation from released or locally trained checkpoints is documented in the top-level **`README.md`**.

Related scripts (under `scripts/online/`):

```text
scripts/online/eval_ind_cluster_pop.py   — Ind / Cluster / Pop three-way comparison
scripts/online/eval_population.py        — Population sweep (Figure 4)
scripts/online/eval_perpatient.py        — Per-patient sweep (Figure 4 per-patient)
scripts/online/sweep_eval_perpatient.sh  — Multi-GPU driver for eval_perpatient.py
```
