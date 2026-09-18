# Offline RL Workflow

This document describes the offline RL extension for the MedGym benchmark.

The main benchmark evaluation workflow can be run from released PINN simulator checkpoints, online behavior policy checkpoints, and offline policy checkpoints under:

```text
results/pinn/
results/online/
results/offline/
```

The fixed-dt offline workflow below does **not** require `data/mimic_pinn_v4_filtered.csv`. When `--no_mimic_csv` is used, patient IDs are discovered from `results/pinn/individual/`, provided through `--patient_ids`, or loaded from a patient-list file via `--patient_ids_file`. Initial-state normalization is loaded from `init_state_norm.npy` in each individual PINN directory.

The processed MIMIC-derived CSV is only needed for PINN training or development-only fallback workflows. It is not required for offline dataset collection, offline policy training, offline policy evaluation, or plotting when the released checkpoints and normalization artifacts are available.

The commands below are written as single-line commands using `python` rather than `python3`, so that they can be run on macOS, Linux, Windows PowerShell, and Windows CMD without shell-specific line-continuation syntax.

### Quick review

- **Fixed-dt only** — The offline smoke-test workflow uses `--dt_mode fixdt`, trains on `*_fixdt` tags, and evaluates fixed-dt offline policies only.
- **No MIMIC CSV for offline evaluation** — Use `--no_mimic_csv` for offline dataset collection and offline policy evaluation.
- **Patient-list driven individual runs** — Use `--patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv` to select the individual patient cohort consistently with the online workflow. Manual `--patient_ids ...` can still be used for tiny smoke tests.
- **DQN implementation** — DQN uses the CQL training script with `--cql_alpha 0.0`.
- **Population-transfer evaluation** — Population-trained offline policies are evaluated on individual patient PINNs using `--eval_scope population_transfer` and `--model_tag trpo_pop_fixdt`.
- **YAML configs** — `configs/offline_rl/*.yaml` record the recommended smoke/default/paper hyperparameters and paths. The current offline scripts are CLI-driven, so pass these values through command-line flags unless a wrapper is added later.

---

## Expected inputs

### Patient list

For cohort-level individual runs, use the same pid-only patient list used by the online workflow:

```text
checkpoints-cohort/cohort_1/cohort_1_training.csv
```

The file can contain one patient ID per line or a CSV header such as `pid`, `patient_id`, `icu_id`, `icustay_id`, or `stay_id`. For a quick local smoke test, `--patient_ids 200325 201046 201101` can be used instead of `--patient_ids_file`.

### PINN simulator artifacts

```text
results/pinn/individual/patient_<pid>/
├── pinn.pt
├── scales.npy
└── init_state_norm.npy

results/pinn/population/
└── scales.npy
```

`pinn.pt` is the patient-specific simulator checkpoint. `scales.npy` stores state/action normalization statistics. `init_state_norm.npy` is required for no-MIMIC evaluation because it provides the normalized initial state without reading the MIMIC-derived CSV.

### Online behavior policies

```text
results/online/individual/patient_<pid>/lagrangian_trpo/K20/fixdt/actor.pt
results/online/population/lagrangian_trpo/K20/fixdt/actor.pt
```

These policies are used to collect the offline behavior datasets.

### Offline policy checkpoints

After offline training, models are saved under:

```text
results/offline/policies/individual/<algo>/models/pid_<pid>_fixdt/actor_final.pt
results/offline/policies/population/<algo>/models/trpo_pop_fixdt/actor_final.pt
```

where `<algo>` is one of:

```text
dqn
cql
gcql
```

---

## Step 1: Collect offline datasets

Collect fixed-dt behavior datasets from individual and population Lagrangian TRPO behavior policies:

```bash
python scripts/offline/collect_offline_data.py --no_mimic_csv --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --pinn_dir results/pinn/individual --ind_root results/online/individual --glb_root results/online/population --glb_scales results/pinn/population/scales.npy --save_dir results/offline --behavior_algo lagrangian_trpo --dt_mode fixdt --n_eval 3 --K 20 --action_noise_std 0.05
```

For a tiny smoke test over three example patients, replace `--patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv` with:

```text
--patient_ids 200325 201046 201101
```

This writes:

```text
results/offline/datasets/pid_<pid>_fixdt_dataset.npz
results/offline/datasets/trpo_pop_fixdt_dataset.npz
results/offline/summary/summary_scores.npy
```

---

## Step 2: Train offline policies

Offline training reads only the collected NPZ datasets under `results/offline/datasets/`. It does not require the MIMIC-derived CSV.

For individual training, the scripts can generate `pid_<pid>_fixdt` dataset tags from `--patient_ids_file`. Population training still uses the explicit population dataset tag `trpo_pop_fixdt`.

### CQL

#### Individual CQL

```bash
python scripts/offline/train_cql.py --dataset_dir results/offline/datasets --save_dir results/offline/policies/individual/cql --epochs 5 --batch_size 256 --cql_alpha 5.0 --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --dt_mode fixdt
```

#### Population CQL

```bash
python scripts/offline/train_cql.py --dataset_dir results/offline/datasets --save_dir results/offline/policies/population/cql --epochs 5 --batch_size 256 --cql_alpha 5.0 --tags trpo_pop_fixdt
```

### DQN

DQN is implemented as the same actor-critic offline training pipeline with the conservative CQL penalty disabled.

#### Individual DQN

```bash
python scripts/offline/train_cql.py --dataset_dir results/offline/datasets --save_dir results/offline/policies/individual/dqn --epochs 5 --batch_size 256 --cql_alpha 0.0 --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --dt_mode fixdt
```

#### Population DQN

```bash
python scripts/offline/train_cql.py --dataset_dir results/offline/datasets --save_dir results/offline/policies/population/dqn --epochs 5 --batch_size 256 --cql_alpha 0.0 --tags trpo_pop_fixdt
```

### Guarded CQL

#### Individual GCQL

```bash
python scripts/offline/train_gcql.py --dataset_dir results/offline/datasets --save_dir results/offline/policies/individual/gcql --epochs 5 --batch_size 256 --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --dt_mode fixdt
```

#### Population GCQL

```bash
python scripts/offline/train_gcql.py --dataset_dir results/offline/datasets --save_dir results/offline/policies/population/gcql --epochs 5 --batch_size 256 --tags trpo_pop_fixdt
```

---

## Step 3: Evaluate offline policies

Offline evaluation uses the individual patient PINN simulators and the trained offline policy checkpoints. Use `--eval_scope individual` for per-patient offline policies and `--eval_scope population_transfer` for population-trained offline policies.

The evaluation script infers the method name from `--save_dir` by default. It saves method-specific rollout datasets such as `pid_<pid>_<algo>_fixdt_dataset.npz` and `<algo>_pop_fixdt_dataset.npz`. If needed, the method name can also be set explicitly with `--method_name dqn`, `--method_name cql`, or `--method_name gcql`.

### CQL

#### Individual CQL evaluation

```bash
python scripts/offline/eval_offline.py --no_mimic_csv --eval_scope individual --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --pinn_dir results/pinn/individual --cql_model_dir results/offline/policies/individual/cql/models --glb_scales results/pinn/population/scales.npy --save_dir results/offline/eval/individual/cql --n_eval 3 --K 20
```

#### Population-transfer CQL evaluation

```bash
python scripts/offline/eval_offline.py --no_mimic_csv --eval_scope population_transfer --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --pinn_dir results/pinn/individual --cql_model_dir results/offline/policies/population/cql/models --glb_scales results/pinn/population/scales.npy --save_dir results/offline/eval/population_transfer/cql --n_eval 3 --K 20 --model_tag trpo_pop_fixdt
```

### DQN

#### Individual DQN evaluation

```bash
python scripts/offline/eval_offline.py --no_mimic_csv --eval_scope individual --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --pinn_dir results/pinn/individual --cql_model_dir results/offline/policies/individual/dqn/models --glb_scales results/pinn/population/scales.npy --save_dir results/offline/eval/individual/dqn --n_eval 3 --K 20
```

#### Population-transfer DQN evaluation

```bash
python scripts/offline/eval_offline.py --no_mimic_csv --eval_scope population_transfer --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --pinn_dir results/pinn/individual --cql_model_dir results/offline/policies/population/dqn/models --glb_scales results/pinn/population/scales.npy --save_dir results/offline/eval/population_transfer/dqn --n_eval 3 --K 20 --model_tag trpo_pop_fixdt
```

### Guarded CQL

#### Individual GCQL evaluation

```bash
python scripts/offline/eval_offline.py --no_mimic_csv --eval_scope individual --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --pinn_dir results/pinn/individual --cql_model_dir results/offline/policies/individual/gcql/models --glb_scales results/pinn/population/scales.npy --save_dir results/offline/eval/individual/gcql --n_eval 3 --K 20
```

#### Population-transfer GCQL evaluation

```bash
python scripts/offline/eval_offline.py --no_mimic_csv --eval_scope population_transfer --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv --pinn_dir results/pinn/individual --cql_model_dir results/offline/policies/population/gcql/models --glb_scales results/pinn/population/scales.npy --save_dir results/offline/eval/population_transfer/gcql --n_eval 3 --K 20 --model_tag trpo_pop_fixdt
```

---

## Step 4: Plot Table 7 and Figures 11--14

After collecting datasets, training policies, and running evaluation, generate the offline summary table and figures:

```bash
python scripts/offline/plot_offline_results.py --results_root results/offline --out_dir results/offline/plots --include_ood
```

This writes:

```text
results/offline/plots/offline_summary_table.csv
results/offline/plots/offline_summary_table.tex
results/offline/plots/fig11_gcql_vs_trpolag_fixdt.png
results/offline/plots/fig12_fixed_ind_vs_pop.png
results/offline/plots/fig13_patientwise_fixed.png
results/offline/plots/fig14_ood_score_individual_fixdt.png
results/offline/plots/fig14_ood_score_individual_fixdt.csv
```

Optional diagnostic plots can be generated with:

```bash
python scripts/offline/plot_offline_results.py --results_root results/offline --out_dir results/offline/plots --include_ood --include_state_density --include_diagnostic_method_comparisons
```

---

## Output structure

```text
results/offline/eval/
├── individual/
│   ├── dqn/
│   │   ├── datasets/
│   │   │   └── pid_<pid>_dqn_fixdt_dataset.npz
│   │   └── summary/
│   ├── cql/
│   │   ├── datasets/
│   │   │   └── pid_<pid>_cql_fixdt_dataset.npz
│   │   └── summary/
│   └── gcql/
│       ├── datasets/
│       │   └── pid_<pid>_gcql_fixdt_dataset.npz
│       └── summary/
└── population_transfer/
    ├── dqn/
    │   ├── datasets/
    │   │   └── dqn_pop_fixdt_dataset.npz
    │   └── summary/
    ├── cql/
    │   ├── datasets/
    │   │   └── cql_pop_fixdt_dataset.npz
    │   └── summary/
    └── gcql/
        ├── datasets/
        │   └── gcql_pop_fixdt_dataset.npz
        └── summary/
```

---

## Notes

- The commands above use `python` rather than `python3` and avoid shell-specific line continuation, so they can be run on macOS, Linux, Windows PowerShell, and Windows CMD.
- The commands above use `--patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv` for patient-list driven individual learning/evaluation. For a small smoke test, replace it with `--patient_ids 200325 201046 201101`.
- To reproduce the paper-scale offline results, increase `--n_eval` and training `--epochs` according to `configs/offline_rl/paper.yaml`.
- The released evaluation workflow should not require `data/mimic_pinn_v4_filtered.csv`. If a script needs the processed MIMIC-derived CSV, it should be treated as a PINN-training or development-only fallback workflow.
