"""
build_filtered_patients.py
===========================
Read ``data/mimic_pinn_v4_filtered.csv``, filter patients by quality rules suitable
for PINN training, and write a single-column ``pid`` CSV in the same layout as
``checkpoints-cohort/cohort_1/cohort_1_training.csv``.

Filtering rules (unchanged from prior convention):
  - Row count              : 50–400 rows
  - ICU time span          : ≥ 24 hours
  - Temporal coverage      : observations in ≥ 7/10 decile bins along the stay
  - Bin-count coefficient of variation : ≤ 1.0
  - Per-state valid observation rate   : ≥ 0.30

Usage (repo root)::
    python data/build_filtered_patients.py
    python data/build_filtered_patients.py \\
        --input data/mimic_pinn_v4_filtered.csv \\
        --output data/mimic_filtered.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATE_FEATURES = ["SpO2", "PaO2", "Bilirubin", "GCS", "Urine_Step", "Lactate"]
ZERO_IS_NAN = {"SpO2", "PaO2", "Bilirubin", "GCS"}

MIN_ROWS = 50
MAX_ROWS = 400
MIN_HOUR_SPAN = 24.0
N_BINS = 10
MIN_BINS_COV = 7
MAX_CV_BINS = 1.0
MIN_FEAT_COV = 0.30


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Filter patients from mimic_pinn_v4_filtered.csv.")
    p.add_argument(
        "--input",
        type=Path,
        default=here / "mimic_pinn_v4_filtered.csv",
        help="Wide-format trajectory CSV (icu_id, hours, state/action columns).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=here / "mimic_filtered.csv",
        help="Output path: single column 'pid' (same layout as checkpoints-cohort/cohort_1/cohort_1_training.csv).",
    )
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional prior pid-list CSV (must have a 'pid' column) to print set diff.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    csv_in = args.input
    csv_out = args.output

    if not csv_in.is_file():
        raise SystemExit(f"Input not found: {csv_in}")

    print(f"Loading {csv_in} ...")
    df_all = pd.read_csv(csv_in)
    if "icu_id" not in df_all.columns or "hours" not in df_all.columns:
        raise SystemExit(f"Expected columns 'icu_id' and 'hours' in {csv_in}")

    # Normalise icu_id to int (CSV may store 200014.0)
    df_all = df_all.copy()
    df_all["icu_id"] = df_all["icu_id"].astype(np.int64)

    print(f"  Total rows: {len(df_all):,}  |  Unique patients: {df_all['icu_id'].nunique():,}")

    records = []
    for pid, pdf in df_all.groupby("icu_id", sort=True):
        pdf = pdf.sort_values("hours")
        hours = pdf["hours"].values.astype(float)
        n_rows = len(pdf)
        hour_span = float(hours[-1] - hours[0]) if n_rows else 0.0

        dts = np.diff(hours)
        if len(dts) == 0:
            continue
        dt_mean = float(dts.mean())
        dt_std = float(dts.std())
        dt_max = float(dts.max())

        if hour_span <= 0:
            continue
        bin_edges = np.linspace(hours[0], hours[-1], N_BINS + 1)
        bin_counts = np.array(
            [((hours >= bin_edges[i]) & (hours < bin_edges[i + 1])).sum() for i in range(N_BINS)]
        )
        bin_counts[-1] += (hours == bin_edges[-1]).sum()
        bins_covered = int((bin_counts > 0).sum())
        cv_bins = float(bin_counts.std() / (bin_counts.mean() + 1e-9))

        cov: dict[str, float] = {}
        for feat in STATE_FEATURES:
            if feat not in pdf.columns:
                cov[feat] = 0.0
                continue
            vals = pdf[feat].values.astype(float)
            if feat in ZERO_IS_NAN:
                valid = (~np.isnan(vals)) & (vals != 0)
            else:
                valid = ~np.isnan(vals)
            cov[feat] = float(valid.mean())

        row: dict = {
            "icu_id": int(pid),
            "n_rows": n_rows,
            "hour_span": round(hour_span, 2),
            "dt_mean": round(dt_mean, 3),
            "dt_std": round(dt_std, 3),
            "dt_max": round(dt_max, 3),
            "bins_covered": bins_covered,
            "cv_bins": round(cv_bins, 3),
        }
        for f in STATE_FEATURES:
            row[f"cov_{f}"] = round(cov[f], 3)
            row[f"_raw_cov_{f}"] = cov[f]
        records.append(row)

    df_stats = pd.DataFrame(records)
    print(f"  Computed stats for {len(df_stats):,} patients")

    mask = (
        (df_stats["n_rows"] >= MIN_ROWS)
        & (df_stats["n_rows"] <= MAX_ROWS)
        & (df_stats["hour_span"] >= MIN_HOUR_SPAN)
        & (df_stats["bins_covered"] >= MIN_BINS_COV)
        & (df_stats["cv_bins"] <= MAX_CV_BINS)
    )
    for feat in STATE_FEATURES:
        mask &= df_stats[f"_raw_cov_{feat}"] >= MIN_FEAT_COV

    df_filtered = df_stats[mask].reset_index(drop=True)

    print(f"\nFiltered: {len(df_stats):,} → {len(df_filtered):,} patients")
    print(f"  n_rows       : {df_filtered.n_rows.min()} – {df_filtered.n_rows.max()}")
    print(f"  hour_span    : {df_filtered.hour_span.min():.1f} – {df_filtered.hour_span.max():.1f} h")
    print(f"  bins_covered : {df_filtered.bins_covered.min()} – {df_filtered.bins_covered.max()}")
    print(f"  cv_bins      : {df_filtered.cv_bins.min():.3f} – {df_filtered.cv_bins.max():.3f}")

    # population.csv layout: header ``pid`` only, one id per row (sorted ascending)
    pids = sorted(int(x) for x in df_filtered["icu_id"].tolist())
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pid": pids}).to_csv(csv_out, index=False)
    print(f"\nSaved (population-style): {csv_out}  ({len(pids)} rows + header)")

    if args.compare is not None and args.compare.is_file():
        df_old = pd.read_csv(args.compare)
        if "pid" not in df_old.columns:
            print(f"[compare] skip: no 'pid' column in {args.compare}")
        else:
            old_ids = set(int(x) for x in df_old["pid"].dropna().tolist())
            new_ids = set(pids)
            print(f"\n=== compare with {args.compare} ===")
            print(f"  old: {len(old_ids):,}  new: {len(new_ids):,}")
            print(f"  intersection: {len(new_ids & old_ids):,}")
            print(f"  new only:     {len(new_ids - old_ids):,}")
            print(f"  old only:     {len(old_ids - new_ids):,}")


if __name__ == "__main__":
    main()
