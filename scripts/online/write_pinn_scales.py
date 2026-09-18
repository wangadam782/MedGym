#!/usr/bin/env python3
"""Write ``scales.npy`` and ``init_state_norm.npy`` from the MIMIC CSV only.

Matches ``scripts/train_pinn.py``: ``load_patients`` → ``enforce_pinn_physical_bounds``
→ ``save_scales`` → first ICU timestep clipped to ``state_min`` / ``state_max`` in
the saved ``scales.npy`` (PINN-normalised space). Does not train ``pinn.pt``.

Run from repo root::

    python3 scripts/online/write_pinn_scales.py \\
        --pinn-root results/pinn_test/individual --force

    python3 scripts/online/write_pinn_scales.py \\
        --csv data/mimic_pinn_v4_filtered.csv \\
        --patient-ids 200325 201046 \\
        --pinn-root results/pinn_test/individual
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent.parent))

from data.config import state_dim
from data.loader import load_patients
from train.pinn.io_utils import enforce_pinn_physical_bounds, save_scales


def _parse_pid(dir_name: str) -> int | None:
    if dir_name.startswith("patient_"):
        try:
            return int(dir_name[len("patient_") :])
        except ValueError:
            return None
    if dir_name.isdigit():
        return int(dir_name)
    return None


def _discover_patient_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _parse_pid(d.name) is not None:
            out.append(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write scales.npy + init_state_norm.npy from CSV (same as train_pinn pretrain I/O)."
    )
    p.add_argument(
        "--pinn-root",
        type=Path,
        default=_REPO / "results" / "pinn_test" / "individual",
        help="Root containing per-patient folders (numeric or patient_<id>).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=_REPO / "data" / "mimic_pinn_v4_filtered.csv",
    )
    p.add_argument(
        "--patient-ids",
        type=int,
        nargs="*",
        default=None,
        help="If set, only these ICU IDs (folders must exist under --pinn-root).",
    )
    p.add_argument(
        "--patient-ids-file",
        type=Path,
        default=None,
        help="Optional text file: one ICU id per line (same convention as sweep scripts).",
    )
    p.add_argument(
        "--all-dirs-ignore-csv-filter",
        action="store_true",
        help="Visit every numeric/patient_* folder under --pinn-root even if icu_id may be absent "
        "from CSV (many failures). Default: only dirs whose id appears in CSV.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scales.npy / init_state_norm.npy.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = p.parse_args()

    root: Path = args.pinn_root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    print(f"Loading {args.csv} (once) ...", flush=True)
    df_all = pd.read_csv(args.csv)

    csv_icu_ids = set(int(x) for x in df_all["icu_id"].unique())

    def _resolve_targets() -> list[Path]:
        if args.patient_ids_file is not None:
            if not args.patient_ids_file.is_file():
                raise SystemExit(f"--patient-ids-file not found: {args.patient_ids_file}")
            raw = args.patient_ids_file.read_text().splitlines()
            pids: list[int] = []
            for line in raw:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tok = line.split()[0]
                if tok.isdigit():
                    pids.append(int(tok))
            targets_loc: list[Path] = []
            for pid in sorted(set(pids)):
                hit = next(
                    (c for c in (root / str(pid), root / f"patient_{pid}") if c.is_dir()),
                    None,
                )
                if hit is None:
                    print(f"[skip] no folder for pid={pid} under {root}", flush=True)
                    continue
                targets_loc.append(hit)
            return targets_loc

        if args.patient_ids is not None:
            out = []
            for pid in args.patient_ids:
                hit = next(
                    (c for c in (root / str(pid), root / f"patient_{pid}") if c.is_dir()),
                    None,
                )
                if hit is None:
                    print(f"[skip] no folder for pid={pid} under {root}", flush=True)
                    continue
                out.append(hit)
            return out

        discovered = _discover_patient_dirs(root)
        if args.all_dirs_ignore_csv_filter:
            return discovered
        return [
            d
            for d in discovered
            if (pid := _parse_pid(d.name)) is not None and pid in csv_icu_ids
        ]

    targets = _resolve_targets()

    written = 0
    skipped = 0
    failed = 0

    for out_dir in targets:
        pid = _parse_pid(out_dir.name)
        if pid is None:
            skipped += 1
            continue

        scales_path = out_dir / "scales.npy"
        init_path = out_dir / "init_state_norm.npy"
        if not args.force and scales_path.exists() and init_path.exists():
            skipped += 1
            continue

        try:
            patient, _, scales = load_patients(str(args.csv), pid, df_all=df_all)
        except Exception as exc:
            print(f"[fail] pid={pid} load_patients: {exc}", flush=True)
            failed += 1
            continue

        scales = enforce_pinn_physical_bounds(scales)

        if args.dry_run:
            print(f"[dry-run] would write {scales_path} {init_path}  pid={pid}", flush=True)
            written += 1
            continue

        save_scales(scales, scales_path)

        sc_np = np.load(str(scales_path), allow_pickle=True).item()
        x0 = patient["data"][0, 1 : 1 + state_dim].detach().cpu().numpy().astype(np.float32)
        x0 = np.clip(
            x0,
            sc_np["state_min"].astype(np.float32),
            sc_np["state_max"].astype(np.float32),
        ).astype(np.float32)
        np.save(str(init_path), x0)
        print(f"[write] pid={pid}  {scales_path.name} + {init_path.name}", flush=True)
        written += 1

    print(
        f"\nDone. written={written} skipped={skipped} failed={failed} root={root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
