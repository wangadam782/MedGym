# PINN Simulator Training

This document describes optional PINN simulator construction.

The main MedGym evaluation workflow uses released PINN checkpoints under:

```text
results/pinn/
  population/
  individual/
```

Therefore, retraining PINN simulators is **not required** for evaluating the released benchmark. The scripts below are provided for transparency and for users who want to construct new simulator benchmarks from their own local clinical datasets.

---

## Data Requirement

PINN training requires a user-provided preprocessed clinical time-series CSV, typically referred to as:

```text
data/mimic_pinn_v4_filtered.csv
```

This file contains MIMIC-III-derived patient-level clinical time-series data and is not redistributed with this repository.

See `data/README.md` for the expected schema and data restrictions.

Users can customize which patients are used to train the population-level PINN and which patients are used to train individual PINNs.

---

## Population PINN Training

Train one shared population PINN simulator:

```bash
python scripts/train_pinn_population.py --csv data/mimic_pinn_v4_filtered.csv --patient_ids_csv checkpoints-cohort/cohort_1/cohort_1_training.csv --save_dir results/pinn/population
```

If you have a population-level patient list, for example `checkpoints-cohort/cohort_1/cohort_1_training.csv`, you can restrict training to those patients by running:

```bash
python scripts/train_pinn_population.py --patient_ids_csv checkpoints-cohort/cohort_1/cohort_1_training.csv --save_dir results/pinn/population
```

In this case, the population PINN is trained only on the patients listed in `checkpoints-cohort/cohort_1/cohort_1_training.csv`.

Expected output:

```text
results/pinn/population/
  pinn.pt
  scales.npy
  _meta.json
```

Additional outputs may include:

```text
medical_ode.pt
neural_ode.pt
patient_ids.txt
loss_history.json
pinn_loss_history.png
patient_fit_plots/
```

---

## Individual PINN Training

Train individual PINNs for a list of patients:

```bash
PATIENT_IDS_FILE=results/paper/cohort/patient_ids.txt SAVE_ROOT=results/pinn/individual N_GPUS=4 JOBS_PER_GPU=2 SKIP_DONE=1 ./scripts/sweep_pinn.sh
```

This script trains one individual PINN per patient and can run multiple training jobs in parallel across GPUs.
Expected output:

```text
results/pinn/individual/
  patient_<pid>/
    pinn.pt
    scales.npy
    _meta.json
```

## Notes

- PINN simulator training is optional for released benchmark evaluation.
- Exact reconstruction of the released PINN checkpoints from MIMIC-III is not part of the main artifact workflow because MIMIC-III-derived patient time-series data cannot be redistributed.
- Users may edit the PINN training scripts to construct new simulator benchmarks from their own local datasets.
