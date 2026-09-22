## Overview

This document describes the patient selection pipeline used to construct the final population cohort for MedGym experiments.

The pipeline is divided into two stages:

1. **Patient quality filtering** from the preprocessed clinical time-series CSV.
2. **Population cohort construction** from patients with available individual PINN simulators.

The first stage requires the preprocessed clinical time-series file:

```text
data/mimic_pinn_v4_filtered.csv
```

This file is required because the filtering step evaluates trajectory length, ICU observation duration, temporal coverage, and observation rates of physiological variables. The filtering step produces a lightweight patient ID list. The second stage then uses this patient ID list together with existing individual PINN simulator checkpoints.

This separation is important because the initial quality filtering produced approximately 2,400 eligible patients, while individual PINN simulators were available for 831 patients due to computational cost. Therefore, the final population cohort is selected from the simulator-available subset rather than directly from all eligible patients.

---

## Pipeline Summary

```text
data/mimic_pinn_v4_filtered.csv
        |
        |  Step 1: trajectory-quality filtering
        v
data/mimic_filtered.csv
        |
        |  Step 2: intersect with existing individual PINN checkpoints
        v
simulator-available patients
        |
        |  Step 3: PINN-to-PINN distance computation
        v
results/pinn_pairwise_dist/distance_matrix.npy
        |
        |  Step 4: tight group construction
        v
results/tight_groups/
        |
        |  Step 5: union of selected patient groups
        v
checkpoints-cohort/cohort_1/cohort_1_training.csv
```

Steps 2–5 in this diagram (intersection with trained individual PINNs, pairwise simulator distances, tight-group construction, and export of the union cohort) are run **in one pass** by `scripts/data_processing/build_population_from_existing_pinns.py`. You do not need to invoke the pairwise-distance or tight-group scripts separately for the standard cohort pipeline; this document still explains each algorithm’s **role** below.

The final output is a one-column CSV:

```csv
pid
200325
201046
201101
201201
201299
201695
202028
202069
...
```

---

## Stage 1: Patient Quality Filtering

The first stage applies trajectory-level quality criteria to the preprocessed clinical time-series CSV. It is implemented in `data/build_filtered_patients.py`.

Input:

```text
data/mimic_pinn_v4_filtered.csv
```

Output:

```text
data/mimic_filtered.csv
```

The output contains a single column, `pid`, listing patients that satisfy the trajectory-quality criteria.

The filtering criteria are based on:

- sufficient number of time-series rows,
- sufficient ICU observation duration,
- adequate temporal coverage,
- stable temporal distribution of observations,
- sufficient observation rate for key physiological variables.

This stage constructs the **eligible patient pool**. In our setting, this step yielded approximately 2,400 patients.

---

## Stage 2: Population Construction from Existing Individual PINNs

This stage is the **integrated entry point** for the cohort used in PINN-distance and tight-group selection: one run of `build_population_from_existing_pinns.py` performs eligibility intersection, optional persistence of intermediate CSVs, pairwise PINN rollout distances, tight-group construction, and writing the final pid-only cohort CSV.

The script starts from the filtered eligible patients and keeps only those with an individual PINN checkpoint under `--patient-dir` (typically `pinn.pt` next to `scales.npy`). If a patient is missing, train their simulator first; see `**docs/pinn.md`** for individual and population PINN training entry points and data requirements.

Example (full pipeline in one invocation):

```bash
python scripts/data_processing/build_population_from_existing_pinns.py --eligible-pids data/mimic_filtered.csv --patient-dir results/pinn/individual --csv data/mimic_pinn_v4_filtered.csv --persist-results --output data/population_paper.csv
```

If `--csv` is unavailable, you must prepare pairwise artifacts in advance (for example, download `results/pinn_pairwise_dist/` generated on another machine) and reuse them with `--skip-pairwise`.

Before running the command below, download and extract `pinn_pairwise_dist.tar.zst` into `results/` so that `results/pinn_pairwise_dist/` already exists:

```bash
cd results
huggingface-cli download anonymous4514/medgym-ICLR2027 pinn_pairwise_dist.tar.zst --repo-type dataset --local-dir .
tar --zstd -xf pinn_pairwise_dist.tar.zst
cd ..
```

Then run:

```bash
python scripts/data_processing/build_population_from_existing_pinns.py --eligible-pids data/mimic_filtered.csv --patient-dir results/pinn/individual --persist-results --skip-pairwise --pairwise-dir results/pinn_pairwise_dist --output data/population_paper.csv
```

In this CSV-missing workflow, `results/pinn_pairwise_dist/` must already exist before running the command above.

If individual PINN checkpoints live outside `results/pinn/individual`, pass `--patient-dir` to that root (directories named `<pid>` or `patient_<pid>` with `pinn.pt`).

`--csv` is required when running the pairwise-distance step. If `--csv` is missing and you still want to continue from existing artifacts, reuse a previously generated pairwise directory with `--skip-pairwise` (and optionally `--skip-tight-groups` if you also want to reuse tight-group outputs).

By default this script writes intermediate artifacts to a **temporary** directory and deletes them when the run finishes; only the file given by `--output` is kept. Add `**--persist-results`** to retain `results/population_selection/`, `results/pinn_pairwise_dist/`, and `results/tight_groups/` (paths can be overridden with `--work-dir`, `--pairwise-dir`, `--tight-groups-dir`).

This stage performs the following operations:

1. Loads the eligible patient list from `--eligible-pids` (for example `data/mimic_filtered.csv`).
2. Scans the individual PINN directory and detects patients with an available `pinn.pt`.
3. Intersects the eligible patient list with the available PINN checkpoint list.
4. Writes the simulator-available patient list.
5. Creates a filtered clinical CSV containing only simulator-available patients.
6. Computes pairwise distances between individual PINN simulators.
7. Constructs tight patient groups of fixed size.
8. Exports the union of patients in the tight groups as the final population cohort.

With `**--persist-results**`, intermediate CSVs and `summary.json` are written under `--work-dir` (default):

```text
results/population_selection/
```

Typical intermediate outputs include:

```text
results/population_selection/simulator_available_patients.csv
results/population_selection/mimic_simulator_available.csv
results/population_selection/summary.json
```

The final output is the path passed to `**--output**` (default `data/population_paper.csv`):

```text
data/population_paer.csv
```

The result should be the same as `checkpoints-cohort/cohort_1/cohort_1_training.csv`

---

## PINN-to-PINN Distance Computation (role)

This step uses **individual PINN simulators**, not raw clinical features alone. For each simulator-available patient, the model is rolled out from patient-specific initial conditions with patient-specific treatment trajectories and observation intervals; pairwise distances summarize **rollout disagreement** between pairs of simulators. That distance is interpreted as **similarity in learned simulator behavior**, which differs from clustering on static clinical tables: two patients are “close” when their trained simulators behave similarly under the same style of clinically grounded rollouts.

When `**--persist-results`** is used, a distance matrix and related diagnostics are written under `results/pinn_pairwise_dist/` (exact filenames may include neighbor lists and optional plots). The integrated script runs this step internally; no separate command is required for the default cohort workflow.

---

## Tight Group Construction (role)

Given the pairwise simulator-distance structure, **tight groups** are fixed-size sets of patients who are mutually close under that metric. The procedure searches for such groups (subject to hyperparameters such as group size). The **final cohort** is the union of all patients appearing in the selected tight groups.

In the repository’s standard pipeline, this grouping step is **invoked from inside** `build_population_from_existing_pinns.py` immediately after the distance step; you do not run a separate tight-group command for the end-to-end cohort build.

---

## Rationale for This Selection Strategy

The goal of this cohort construction procedure is not to randomly sample patients from the clinical dataset. Instead, the objective is to obtain a patient cohort that is suitable for evaluating both population-level and individual-level PINN/RL workflows.

The selection strategy is motivated by three considerations.

### 1. Quality Control Before Simulator Training

PINN training requires sufficiently informative longitudinal trajectories. Patients with very short, sparse, or irregularly distributed observations may lead to unreliable simulator training. Therefore, the first filtering stage removes patients that do not satisfy basic trajectory-quality requirements.

This produces an eligible pool of patients that are appropriate candidates for simulator training.

### 2. Use of Simulator-Available Patients

Although approximately 2,400 patients passed the quality filter, training individual PINN simulators for all of them was computationally expensive. In the current experimental setup, individual PINN simulators were available for 831 patients after preprocessing.

Therefore, subsequent simulator-distance computation and patient selection were performed only on these 831 simulator-available patients.

This distinction is important:

```text
eligible patients after quality filtering: approximately 2,400
patients with successfully trained individual PINN simulators: 831
patients used for PINN-based cohort selection: 831
```

The final population cohort is thus selected from the 831 patients for whom individual simulator behavior could be evaluated.

### 3. Simulator-Behavior-Based Cohort Construction

The final patient groups are selected using distances between trained individual PINN simulators. This makes the selection based on learned patient dynamics rather than only on raw clinical measurements or hand-crafted features.

This is useful because the downstream benchmark evaluates simulator-based decision making and policy learning. Selecting patients using simulator-behavior similarity makes the cohort construction more aligned with the actual modeling and evaluation pipeline.

### 4. Diverse but Non-Outlier Simulator Selection

The tight-group construction was used to obtain a population cohort that is diverse across patient simulator behaviors while avoiding isolated outlier simulators. A patient simulator that is far from most others may reflect rare physiology, unstable simulator training, or insufficiently reliable learned dynamics. Directly sampling patients from the simulator-available cohort could therefore include such isolated cases.
By contrast, tight groups are constructed to be both internally coherent and mutually separated: each group must satisfy a small within-group mean distance, and each new group center must be at least `min_center_dist` away from previously selected centers. Thus, the resulting cohort is intended to be representative, diverse, and stable for subsequent population-level and individual-level PINN/RL evaluation.
---

## Important Notes

- `data/mimic_pinn_v4_filtered.csv` is required for the preprocessing stage.
- `build_population_from_existing_pinns.py` requires `--csv` when generating pairwise distances; there is no CSV-free fallback based on `init_state_norm.npy` / `scales.npy`.
- The 831 patients should be described as the **simulator-available cohort**, not as an arbitrary manually selected subset.
- The final cohort is selected from patients with available individual PINN simulators.
- The tight groups are based on learned PINN rollout behavior.
- Heavy intermediate artifacts such as distance matrices and plots should be stored under `results/`.
- Lightweight patient ID lists such as `mimic_filtered.csv` (eligible pool) and `population.csv` (final cohort) can be stored under `data/`.
- **`checkpoints-cohort/cohort_1/cohort_1_training.csv` is not fixed:** you may edit it (same one-column `pid` layout), pass another pid-only CSV via flags such as `--patient_ids_csv` in `scripts/train_pinn_population.py`, or **replace this pipeline entirely** with your own cohort algorithm and write the same format. Lines starting with `#` in that CSV are ignored when loading. See **`data/README.md`** (end section on `population.csv`).

