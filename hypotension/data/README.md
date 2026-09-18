# Hypotension Data

## Data Access

This repository does **not** redistribute the Health Gym dataset.

The hypotension benchmark uses the **PhysioNet Health Gym — Synthetic Acute Hypotension v1.0.0** dataset:

> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
> *Synthetic Patient Data in Health*. PhysioNet.
> https://physionet.org/content/synthetic-mimic-iii-health-gym/1.0.0/

Access requires a PhysioNet account and agreement to the data use terms. After approval:

```bash
wget -r -N -c -np --user <your-physionet-username> --ask-password \
    https://physionet.org/files/synthetic-mimic-iii-health-gym/1.0.0/
```

Place the downloaded `hypotension.csv` (or the equivalent acute-hypotension file) at:

```text
hypotension/data/hypotension_healthgym.csv
```

## Preprocessing

Run once from the medrl-tacos root after placing the CSV:

```bash
python -m hypotension.data.preprocess
```

Outputs in `hypotension/data/processed/` (gitignored):

```text
processed/
  scales.npy          {mean, std, ascl_mean, ascl_std, state_min, state_max}
  patients/P####.npz  t, X (physical units), U (physical units), M (obs mask)
  cohort_long.csv     cleaned long-format table
```

## Column mapping

The preprocessor reads the following columns from the CSV (see `RAW` dict in `preprocess.py`):

| Raw column | Used as |
|---|---|
| `PatientID` | patient ID |
| `Timepoints` | time (hours) |
| `MAP`, `systolic_bp`, `diastolic_bp` | MAP, PP |
| `urine` | urine output |
| `ALT`, `AST` | hepatic marker |
| `PO2` | PaO2 |
| `lactic_acid`, `serum_creatinine` | Lac, Cr |
| `fluid_boluses`, `vasopressors`, `FiO2`, `GCS_total` | actions + GCS |

To adapt to a different schema, edit the `RAW` mapping at the top of `preprocess.py`.
