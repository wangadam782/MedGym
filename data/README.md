> **Data files not included**: `mimic_pinn_v4_filtered.csv` and related patient CSV files
> contain MIMIC-IV-derived clinical data and are not redistributed. See MIMIC-IV data use
> agreement for access. Patient cohort split files are in `cohort_splits/cohort_1/`.

# Data

This directory contains configuration, cohort metadata, and data loading
utilities only. 

---

## Data Source

This project uses the
[MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/).
MIMIC-III is a restricted-access resource. Redistribution of any MIMIC-derived
data is strictly prohibited.
---

## Preprocessing Pipeline

1. Follow the official instructions to obtain access to the MIMIC-III v1.4
   clinical dataset from PhysioNet:
   [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)

2. Clone and follow the instructions from the official GitHub repository:
   [microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis)
   Use the provided SQL scripts and Python code to extract intermediate
   tables for the sepsis cohort.

3. Replace `sepsis_cohort.py` in the cloned repository with
   `sepsis_cohort_step.py` provided in this project.

---

## Files in This Directory

| File                    | Description                                                                               |
|-------------------------|-------------------------------------------------------------------------------------------|
| `population.csv`        | Anonymous `icustay_id` list defining the study cohort                                     |
| `config.py`             | Dimension constants, normalization statistics, device config                              |
| `loader.py`             | PyTorch dataset and dataloader utilities                                                  |
| `sepsis_cohort_step.py` | This generates a dataset of individually timestamped observations from the sepsis cohort. |
| `build_filtered_patients.py`  | Filters high-quality patient trajectories for PINN training and outputs a population-style `pid` list |
