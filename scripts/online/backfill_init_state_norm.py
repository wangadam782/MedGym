#!/usr/bin/env python3
"""Write ``init_state_norm.npy`` next to each individual ``scales.npy``.

Uses the same mapping as eval scripts: first CSV timestep in ``load_patients``
coordinates → clip to bounds from ``load_pinn_folder_scales`` (``scales.npy``).

To **rebuild both** ``scales.npy`` and ``init_state_norm.npy`` from the MIMIC CSV
using the *same* logic as ``scripts/train_pinn.py`` (no training), use
``scripts/online/write_pinn_scales.py`` instead.

Run from repo root::

    python scripts/online/backfill_init_state_norm.py
    python scripts/online/backfill_init_state_norm.py --pinn-root results/pinn/individual --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from data.config import state_dim
from data.loader import load_patients
from utils.pinn_scales import (
    INIT_STATE_NORM_FILENAME,
    csv_norm_state_to_pinn_norm_state,
    load_pinn_folder_scales,
    save_init_state_norm,
)

PHYSICAL_MIN = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32)
PHYSICAL_MAX = np.array([100.0, 600.0, 30.0, 15.0, 5000.0, 20.0], dtype=np.float32)


def _parse_pid(dir_name: str) -> int | None:
    if dir_name.startswith("patient_"):
        try:
            return int(dir_name[len("patient_") :])
        except ValueError:
            return None
    if dir_name.isdigit():
        return int(dir_name)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pinn-root",
        type=Path,
        default=_REPO / "results" / "pinn" / "individual",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=_REPO / "data" / "mimic_pinn_v4_filtered.csv",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing init_state_norm.npy.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only.",
    )
    args = p.parse_args()

    root: Path = args.pinn_root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    print(f"Loading {args.csv} (once) ...", flush=True)
    df_all = pd.read_csv(args.csv)

    written = 0
    skipped = 0
    failed = 0

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        scales_ok = (d / "scales.npy").exists() or (d / "cluster_scales.npy").exists()
        if not scales_ok:
            continue

        out_path = d / INIT_STATE_NORM_FILENAME
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        pid = _parse_pid(d.name)
        if pid is None:
            print(f"[skip] cannot parse patient id from dir name: {d.name}", flush=True)
            skipped += 1
            continue

        try:
            patient, _, scales = load_patients(str(args.csv), pid, df_all=df_all)
        except Exception as exc:
            print(f"[fail] pid={pid} load_patients: {exc}", flush=True)
            failed += 1
            continue

        try:
            mean_npy, std_npy, _, smin_npy, smax_npy, _ = load_pinn_folder_scales(
                d, PHYSICAL_MIN, PHYSICAL_MAX
            )
        except FileNotFoundError as exc:
            print(f"[fail] pid={pid} scales: {exc}", flush=True)
            failed += 1
            continue

        mean_csv = scales["state_mean"].cpu().numpy().astype(np.float32)
        std_csv = scales["state_std"].cpu().numpy().astype(np.float32)
        row0 = patient["data"][0, 1 : 1 + state_dim].cpu().numpy().astype(np.float32)
        init_norm = csv_norm_state_to_pinn_norm_state(
            row0, mean_csv, std_csv, mean_npy, std_npy, smin_npy, smax_npy
        )

        if args.dry_run:
            print(f"[dry-run] would write {out_path}  pid={pid}", flush=True)
        else:
            save_init_state_norm(out_path, init_norm)
            print(f"[write] {out_path}  pid={pid}", flush=True)
        written += 1

    print(
        f"\nDone. written={written} skipped_existing={skipped} failed={failed} "
        f"root={root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
