# Full Pipeline Reproduction

Stage scripts in `scripts/` cover the complete workflow. All commands run from the repository root. Pre-trained checkpoints are on HuggingFace for stages 2–4; stages 0–1 require the MIMIC-IV trajectory CSV.

## Prerequisites

```bash
uv sync
```

Place the DUA-compliant MIMIC-IV file at `data/mimic_pinn_v4_filtered.csv` before running stages 0–1.

---

## Stage 0 — Patient Clustering

```bash
COHORT=cohort_1 bash scripts/clustering.sh
```

Outputs `cohort_splits/cohort_1/cluster_map.csv`. Key variables: `GROUP_SIZE` (default 11), `MAX_GROUPS` (default 10).

---

## Stage 1 — PINN Training

```bash
COHORT=cohort_1 bash scripts/pinn.sh              # all scopes
COHORT=cohort_1 SCOPE=individual bash scripts/pinn.sh
COHORT=cohort_1 SCOPE=cluster    bash scripts/pinn.sh
COHORT=cohort_1 SCOPE=population bash scripts/pinn.sh
```

---

## Stage 2 — Online RL Training

```bash
COHORT=cohort_1 ALGOS="sac ppo trpo lagrangian_ppo lagrangian_trpo" \
    bash scripts/online_train.sh
```

---

## Stage 3 — Online RL Evaluation

```bash
for ALGO in sac ppo trpo lagrangian_ppo lagrangian_trpo; do
    COHORT=cohort_1 ALGO=$ALGO N_EVAL=50 bash scripts/online_eval.sh
done
```

---

## Stage 4 — Offline RL

```bash
COHORT=cohort_1 EPOCHS=4000 bash scripts/offline.sh
```

---

## All Cohorts

```bash
for N in 1 2 3 4 5 6 7; do
    COHORT=cohort_${N} SKIP_DONE=1 bash scripts/online_eval.sh
    COHORT=cohort_${N} SKIP_DONE=1 bash scripts/offline.sh
done
```

---

## Dry Run

```bash
DRY_RUN=1 COHORT=cohort_1 bash scripts/clustering.sh
```
