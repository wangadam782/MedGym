# HuggingFace Data Setup

This document describes how to download the released checkpoints from HuggingFace and use them with this repository.

## Dataset Access

The sepsis checkpoint dataset is gated to comply with MIMIC-IV data use agreements.
Request access at the HuggingFace dataset repository (see README for the current repo ID).

After approval, authenticate:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

---

## Downloading Checkpoints

Checkpoints are released as two `.tar.zst` archives.

### checkpoints-cohort (~2.1 GB)

PINN and online/offline policy checkpoints for 7 cohorts (cohort_1 – cohort_7).
Cohort_1 also includes pre-trained offline policies (Table 3).

```bash
huggingface-cli download <HF_REPO_ID> \
    checkpoints-cohort.tar.zst --repo-type dataset --local-dir .
tar --zstd -xf checkpoints-cohort.tar.zst
```

### checkpoints-remaining15 (~450 MB)

PINN and policy checkpoints for the 15-patient medoid/nearest clustering experiments,
plus pre-trained offline policies (Table 12).

```bash
huggingface-cli download <HF_REPO_ID> \
    checkpoints-remaining15.tar.zst --repo-type dataset --local-dir .
tar --zstd -xf checkpoints-remaining15.tar.zst
```

---

## Directory Layout After Download

### checkpoints-cohort

```text
checkpoints-cohort/
  cohort_1/ ... cohort_7/
    cohort_N_training.csv          ← 110 training patient IDs
    cohort_N_test.csv              ← cluster/split table
    pinn/
      ind/patient_<pid>/           ← individual PINN per test patient
      clu/cluster_<1-10>/          ← cluster-pooled PINN
      pop/                         ← population PINN
    online/
      ind/patient_<pid>/<algo>/K20/<fixdt|vardt>/
      clu/cluster_<1-10>/<algo>/K20/<fixdt|vardt>/
      pop/<algo>/K20/<fixdt|vardt>/
    offline/                       ← (cohort_1 only) pre-trained offline policies
      ind/<dqn|cql|gcql>/models/ind_<pid>_fixdt/actor_final.pt
      clu/<dqn|cql|gcql>/models/cluster<k>_fixdt/actor_final.pt
      pop/<dqn|cql|gcql>/models/pop_fixdt/actor_final.pt
```

### checkpoints-remaining15

```text
checkpoints-remaining15/
  pinn/
    ind/patient_<pid>/             ← 15 medoid patients
    clu/patient_<pid>/             ← 15 nearest patients
    pop/                           ← shared population PINN (all 15 patients)
  online/
    ind/patient_<pid>/<algo>/K20/<fixdt|vardt>/
    clu/patient_<pid>/<algo>/K20/<fixdt|vardt>/
    pop/<algo>/K20/<fixdt|vardt>/
  offline/
    ind/<dqn|cql|gcql>/models/ind_<medoid_pid>_fixdt/   ← medoid individual policies
    clu/<dqn|cql|gcql>/models/clu_<nearest_pid>_fixdt/  ← nearest-patient policies
    pop/<dqn|cql|gcql>/models/pop_fixdt/                ← shared population policy
```

---

## Running Evaluation on Downloaded Checkpoints

### Online evaluation (cohort_1)

```bash
COHORT=cohort_1 bash scripts/online_eval.sh
```

### Offline evaluation with pre-trained policies (cohort_1)

Evaluate the downloaded offline policies directly without retraining:

```bash
COHORT=cohort_1 \
PINN_DIR=checkpoints-cohort/cohort_1/pinn/ind \
POLICY_ROOT=checkpoints-cohort/cohort_1/offline \
EVAL_ROOT=results/offline_eval/cohort_1 \
bash scripts/offline.sh --eval_only
```

This runs only Step 5 (evaluation) using the pre-trained policies from the download.

### Offline evaluation (remaining15)

```bash
# Build a patient → cluster map for remaining15 ind patients
# (ind = medoid patients; clu = nearest patients)
python scripts/offline/eval_cluster_offline.py \
    --cluster_map  cohort_splits/remaining15/cluster_map.csv \
    --policy_root  checkpoints-remaining15/offline \
    --pinn_dir     checkpoints-remaining15/pinn/ind \
    --eval_root    results/offline_eval/remaining15 \
    --clu_by_pid
```

The `--clu_by_pid` flag switches clu model tags from `cluster{k}_fixdt` (cohort format)
to `clu_{pid}_fixdt` (remaining15 format).

---

## Cohort Split Files

Cluster assignment files are committed in `cohort_splits/` and do not require HuggingFace access:

```text
cohort_splits/
  cohort_1/
    cluster_map.csv    ← patient_id → cluster_id mapping (authoritative for cohort_1)
```
