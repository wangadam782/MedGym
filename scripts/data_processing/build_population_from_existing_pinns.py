#!/usr/bin/env python3
"""
build_population_from_existing_pinns.py
=======================================

Build the final population pid CSV from already-trained individual PINN
simulators.

This script intentionally does NOT run patient-quality preprocessing.
Run that separately first, for example:

    python data/build_filtered_patients.py \
        --input data/mimic_pinn_v4_filtered.csv \
        --output data/mimic_filtered.csv

Then run this script (default: intermediate outputs under ``results/`` are written
to a **temporary** directory and removed at the end; only ``--output`` and logs
remain). To keep ``results/population_selection``, ``results/pinn_pairwise_dist``,
and ``results/tight_groups`` on disk, add ``--persist-results``:

    python scripts/data_processing/build_population_from_existing_pinns.py \
        --eligible-pids data/mimic_filtered.csv \
        --patient-dir results/pinn/individual \
        --csv data/mimic_pinn_v4_filtered.csv \
        --persist-results \
        --output checkpoints-cohort/cohort_1/cohort_1_training.csv

If ``--csv`` does not exist on disk (e.g. no ``data/mimic_pinn_v4_filtered.csv``), the
script skips writing the subset trajectory CSV and runs ``compute_pinn_pairwise_dist.py``
with ``--pinn-only``. Patients without ``scales.npy`` / ``cluster_scales.npy`` or without
usable initial conditions (``init_state_norm.npy`` or ``state_min`` / ``state_max`` in
scales) are **dropped** from this run with a log line; the pairwise step only sees the rest.

Pipeline
--------
1. Read eligible patient ids from --eligible-pids.
2. Keep only patients whose individual PINN checkpoint exists under --patient-dir.
3. Write simulator_available_patients.csv and, when ``--csv`` exists, a subset time-series CSV (under work dir).
4. Run compute_pinn_pairwise_dist.py on the simulator-available patients.
5. Run find_tight_groups.py on the pairwise distance output.
6. Union all patient ids appearing in tight groups and write a pid-only CSV to ``--output``.

Unless ``--persist-results`` is set, steps 3–5 use a **temporary** directory tree that
is deleted after a successful or failed run (only ``--output`` remains).

Default paths are resolved relative to the current working directory, so run this
from the repository root.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ID_COLUMNS = ("pid", "icu_id", "patient_id", "stay_id")
MEMBER_COLUMNS = (
    "members", "member", "pids", "patients", "patient_ids", "icu_ids", "ids",
    "group_members", "cluster_members",
)


def repo_root() -> Path:
    # Prefer current working directory. This makes relative paths behave the
    # same whether this file is placed in scripts/ or data/.
    return Path.cwd().resolve()


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_cmd(cmd: list[str], *, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_pid_csv(path: Path) -> list[int]:
    if not path.is_file():
        raise SystemExit(f"PID CSV not found: {path}")
    df = pd.read_csv(path, comment="#")
    for col in ID_COLUMNS:
        if col in df.columns:
            return sorted({int(x) for x in df[col].dropna().tolist()})
    col = df.columns[0]
    return sorted({int(x) for x in df[col].dropna().tolist()})


def write_pid_csv(pids: Iterable[int], path: Path) -> None:
    ids = sorted({int(x) for x in pids})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pid": ids}).to_csv(path, index=False)
    print(f"[write] {path}  ({len(ids)} patients)")


def patient_pinn_exists(patient_dir: Path, pid: int) -> bool:
    return patient_pinn_folder(patient_dir, pid) is not None


def patient_pinn_folder(patient_dir: Path, pid: int) -> Path | None:
    """Directory containing ``pinn.pt`` or ``model.pt`` for ``pid``, if any."""
    for base in (patient_dir / str(pid), patient_dir / f"patient_{pid}"):
        if not base.is_dir():
            continue
        for fname in ("pinn.pt", "model.pt"):
            if (base / fname).is_file():
                return base
    return None


def subset_timeseries_csv(csv_in: Path, ids: set[int], csv_out: Path) -> None:
    if not csv_in.is_file():
        raise SystemExit(f"Input CSV not found: {csv_in}")
    df = pd.read_csv(csv_in)
    if "icu_id" not in df.columns:
        raise SystemExit(f"Expected column 'icu_id' in {csv_in}")
    df = df[df["icu_id"].astype(int).isin(ids)].copy()
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[write] {csv_out}  ({len(df):,} rows, {df['icu_id'].nunique():,} patients)")


def resolve_script(name: str, explicit: str | None, candidates: list[Path]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        p = p.resolve() if p.is_absolute() else (repo_root() / p).resolve()
        if not p.is_file():
            raise SystemExit(f"Script not found: {p}")
        return p
    for p in candidates:
        if p.is_file():
            return p
    tried = "\n  ".join(str(x) for x in candidates)
    raise SystemExit(f"Could not find {name}. Tried:\n  {tried}\nPass --{name.replace('_', '-')}-script explicitly.")


def fill_template_command(template: str, values: dict[str, Any]) -> list[str]:
    rendered = template.format(**{k: str(v) for k, v in values.items()})
    return shlex.split(rendered)


def build_pairwise_cmd(
    *,
    pairwise_script: Path,
    root: Path,
    patient_dir: Path,
    ids_csv: Path,
    timeseries_csv: Path | None,
    pairwise_dir: Path,
    k_steps: int,
    skip_mds: bool,
    skip_plots: bool,
    device: str,
    global_std_path: Path | None,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable, str(pairwise_script),
        "--repo-root", str(root),
        "--patient-dir", str(patient_dir),
        "--patient-ids-csv", str(ids_csv),
        "--out-dir", str(pairwise_dir),
        "--k-steps", str(k_steps),
    ]
    if timeseries_csv is None:
        raise RuntimeError("internal: timeseries_csv is required")
    cmd += ["--csv", str(timeseries_csv)]
    if skip_mds:
        cmd.append("--skip-mds")
    if skip_plots:
        cmd.append("--skip-plots")
    if device != "auto":
        cmd += ["--device", device]
    if global_std_path is not None:
        cmd += ["--global-std-path", str(global_std_path)]
    cmd += extra_args
    return cmd


def ints_from_text(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"(?<!\d)\d{4,}(?!\d)", str(text))]


def collect_ids_from_obj(obj: Any, allowed: set[int]) -> set[int]:
    found: set[int] = set()
    if obj is None or isinstance(obj, bool):
        return found
    if isinstance(obj, int):
        return {obj} if obj in allowed else set()
    if isinstance(obj, float):
        return {int(obj)} if obj.is_integer() and int(obj) in allowed else set()
    if isinstance(obj, str):
        return {x for x in ints_from_text(obj) if x in allowed}
    if isinstance(obj, dict):
        for v in obj.values():
            found.update(collect_ids_from_obj(v, allowed))
        return found
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            found.update(collect_ids_from_obj(v, allowed))
        return found
    return found


def collect_ids_from_csv(path: Path, allowed: set[int]) -> set[int]:
    found: set[int] = set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return found

    for col in ID_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist()
            found.update(x for x in vals if x in allowed)

    for col in MEMBER_COLUMNS:
        if col in df.columns:
            for val in df[col].dropna().tolist():
                found.update(x for x in ints_from_text(str(val)) if x in allowed)

    if not found:
        for col in df.columns:
            if df[col].dtype == object:
                for val in df[col].dropna().tolist():
                    found.update(x for x in ints_from_text(str(val)) if x in allowed)
    return found


def collect_ids_from_json(path: Path, allowed: set[int]) -> set[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return set()
    return collect_ids_from_obj(obj, allowed)


def collect_ids_from_text_file(path: Path, allowed: set[int]) -> set[int]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    return {x for x in ints_from_text(text) if x in allowed}


def looks_like_group_output(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".txt", ".tsv"}:
        return False
    bad = ("pairwise", "distance", "matrix", "dist_matrix", "log", "stats")
    if any(b in name for b in bad):
        return False
    good = ("group", "tight", "cluster", "selected", "population", "pid", "member")
    return any(g in name for g in good) or path.suffix.lower() in {".json", ".jsonl"}


def collect_tight_group_pids(tight_groups_dir: Path, allowed: set[int]) -> tuple[set[int], dict[str, int]]:
    if not tight_groups_dir.is_dir():
        raise SystemExit(f"Tight-groups output directory not found: {tight_groups_dir}")

    files = [p for p in tight_groups_dir.rglob("*") if p.is_file() and looks_like_group_output(p)]
    if not files:
        files = [
            p for p in tight_groups_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".txt", ".tsv"}
        ]

    total: set[int] = set()
    per_file: dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            ids = collect_ids_from_csv(path, allowed)
        elif suffix in {".json", ".jsonl"}:
            ids = collect_ids_from_json(path, allowed)
            if not ids and suffix == ".jsonl":
                ids = collect_ids_from_text_file(path, allowed)
        else:
            ids = collect_ids_from_text_file(path, allowed)
        if ids:
            total.update(ids)
            per_file[str(path)] = len(ids)
    return total, per_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build final population pid CSV from existing individual PINN simulators.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eligible-pids", type=Path, default=Path("data/mimic_filtered.csv"),
                   help="Quality-filtered pid CSV produced by build_filtered_patients.py.")
    p.add_argument("--csv", type=Path, default=Path("data/mimic_pinn_v4_filtered.csv"),
                   help="Full patient time-series CSV used for pairwise distance calculation.")
    p.add_argument("--patient-dir", type=Path, default=Path("results/pinn/individual"),
                   help="Directory containing individual PINN outputs, e.g. <pid>/pinn.pt or patient_<pid>/pinn.pt.")
    p.add_argument("--work-dir", type=Path, default=Path("results/population_selection"),
                   help="With --persist-results: directory for intermediate CSVs and summary.json. "
                   "Ignored when persistence is off (ephemeral temp dir is used).")
    p.add_argument("--pairwise-dir", type=Path, default=Path("results/pinn_pairwise_dist"),
                   help="With --persist-results: output dir for compute_pinn_pairwise_dist.py. "
                   "Ignored when persistence is off.")
    p.add_argument("--tight-groups-dir", type=Path, default=Path("results/tight_groups"),
                   help="With --persist-results: output dir for find_tight_groups.py. "
                   "Ignored when persistence is off.")
    p.add_argument("--output", type=Path, default=Path("checkpoints-cohort/cohort_1/cohort_1_training.csv"),
                   help="Final output CSV. Single column: pid.")
    p.add_argument("--group-size", type=int, default=10,
                   help="Group size passed to find_tight_groups.py.")
    p.add_argument("--k-steps", type=int, default=20,
                   help="Rollout horizon passed to compute_pinn_pairwise_dist.py.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device passed to compute_pinn_pairwise_dist.py.")
    p.add_argument("--skip-mds", action="store_true",
                   help="Pass --skip-mds to compute_pinn_pairwise_dist.py.")
    p.add_argument("--skip-plots", action="store_true",
                   help="Pass --skip-plots to compute_pinn_pairwise_dist.py.")
    p.add_argument("--global-std-path", type=Path, default=None,
                   help="Optional global_std.npy passed to compute_pinn_pairwise_dist.py.")

    p.add_argument("--pairwise-script", type=str, default=None,
                   help="Path to compute_pinn_pairwise_dist.py. Auto-resolved if omitted.")
    p.add_argument("--tight-groups-script", type=str, default=None,
                   help="Path to find_tight_groups.py. Auto-resolved if omitted.")
    p.add_argument("--pairwise-cmd", type=str, default=None,
                   help="Full pairwise command template. Overrides automatic pairwise command construction.")
    p.add_argument("--tight-groups-cmd", type=str, default=None,
                   help="Full tight-groups command template. Overrides default find_tight_groups.py command.")

    p.add_argument("--skip-pairwise", action="store_true",
                   help="Skip pairwise computation and reuse --pairwise-dir.")
    p.add_argument("--skip-tight-groups", action="store_true",
                   help="Skip find_tight_groups.py and reuse --tight-groups-dir.")
    p.add_argument("--no-require-pinn", action="store_true",
                   help="Do not filter to patients with individual pinn.pt/model.pt under --patient-dir.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running them.")
    p.add_argument(
        "--persist-results",
        action="store_true",
        help="Keep --work-dir, --pairwise-dir, and --tight-groups-dir under the repo "
        "(default: use a temp directory and delete it after the run).",
    )
    # REMAINDER must be the last add_argument() so other flags keep correct defaults.
    p.add_argument("--pairwise-extra-args", nargs=argparse.REMAINDER, default=[],
                   help="Extra args appended to the pairwise script. Must be last on the CLI; or use --pairwise-cmd.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    here = script_dir()

    if not args.persist_results and (args.skip_pairwise or args.skip_tight_groups):
        raise SystemExit(
            "--skip-pairwise / --skip-tight-groups require existing on-disk outputs. "
            "Re-run with --persist-results (or run the full pipeline without skip flags)."
        )

    temp_root: Path | None = None
    if args.persist_results:
        work_dir = resolve_path(args.work_dir, root)
        pairwise_dir = resolve_path(args.pairwise_dir, root)
        tight_groups_dir = resolve_path(args.tight_groups_dir, root)
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="medrl_population_build_"))
        work_dir = temp_root / "population_selection"
        pairwise_dir = temp_root / "pinn_pairwise_dist"
        tight_groups_dir = temp_root / "tight_groups"
        print(f"[paths] ephemeral work tree: {temp_root}", flush=True)

    had_ephemeral = temp_root is not None
    try:
        _run_build(
            args,
            root,
            here,
            work_dir=work_dir,
            pairwise_dir=pairwise_dir,
            tight_groups_dir=tight_groups_dir,
        )
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            print(f"[cleanup] removed temp tree {temp_root}", flush=True)
    if had_ephemeral:
        print(
            "[note] Intermediate CSVs, pairwise/tight-group outputs, and summary.json "
            "were under the temp tree above; only --output was kept under the repo.",
            flush=True,
        )


def _run_build(
    args: argparse.Namespace,
    root: Path,
    here: Path,
    *,
    work_dir: Path,
    pairwise_dir: Path,
    tight_groups_dir: Path,
) -> None:
    # Resolve all user-facing paths once. Keep them as Path objects.
    eligible_pids = resolve_path(
        args.eligible_pids if args.eligible_pids is not None else Path("data/mimic_filtered.csv"),
        root,
    )
    csv_path = resolve_path(
        args.csv if args.csv is not None else Path("data/mimic_pinn_v4_filtered.csv"),
        root,
    )
    patient_dir = resolve_path(args.patient_dir, root)
    output = resolve_path(args.output, root)
    global_std_path = resolve_path(args.global_std_path, root) if args.global_std_path else None

    work_dir.mkdir(parents=True, exist_ok=True)
    simulator_available_csv = work_dir / "simulator_available_patients.csv"

    eligible_ids = read_pid_csv(eligible_pids)
    print(f"[ids] eligible patients: {len(eligible_ids)} from {eligible_pids}")

    if args.no_require_pinn:
        usable_ids = eligible_ids
        missing_ids: list[int] = []
    else:
        usable_ids = [pid for pid in eligible_ids if patient_pinn_exists(patient_dir, pid)]
        usable_set = set(usable_ids)
        missing_ids = [pid for pid in eligible_ids if pid not in usable_set]
        print(f"[ids] with individual PINN under {patient_dir}: {len(usable_ids)}")
        if missing_ids:
            print(f"[ids] missing individual PINN: {len(missing_ids)}  (first 20: {missing_ids[:20]})")

    if not usable_ids:
        raise SystemExit("No usable patient ids after filtering. Check --patient-dir or use --no-require-pinn.")

    csv_ok = csv_path.is_file()
    subset_csv: Path | None = work_dir / "mimic_simulator_available.csv"
    if not csv_ok and not args.skip_pairwise:
        raise SystemExit(
            f"CSV not found: {csv_path}. "
            "Pairwise computation requires --csv; re-run with a valid CSV path or "
            "reuse existing pairwise outputs via --skip-pairwise."
        )
    if csv_ok:
        subset_timeseries_csv(csv_path, set(usable_ids), subset_csv)
    else:
        subset_csv = None
        print("[csv] missing, but pairwise step is skipped.", flush=True)

    write_pid_csv(usable_ids, simulator_available_csv)

    values = {
        "python": sys.executable,
        "repo_root": root,
        "csv": csv_path,
        "timeseries_csv": subset_csv if subset_csv is not None else "",
        "ids_csv": simulator_available_csv,
        "patient_dir": patient_dir,
        "pairwise_dir": pairwise_dir,
        "tight_groups_dir": tight_groups_dir,
        "group_size": args.group_size,
        "k_steps": args.k_steps,
        "global_std_path": global_std_path or "",
        "output": output,
    }

    pairwise_script = resolve_script(
        "pairwise_script",
        args.pairwise_script,
        [
            here / "compute_pinn_pairwise_dist.py",
            root / "scripts" / "compute_pinn_pairwise_dist.py",
            root / "data" / "compute_pinn_pairwise_dist.py",
            root / "compute_pinn_pairwise_dist.py",
            here / "compute_pair_pairwise.py",
            root / "scripts" / "compute_pair_pairwise.py",
            root / "data" / "compute_pair_pairwise.py",
            root / "compute_pair_pairwise.py",
            here / "compute_pairwise.py",
            root / "scripts" / "compute_pairwise.py",
            root / "data" / "compute_pairwise.py",
            root / "compute_pairwise.py",
        ],
    )

    if not args.skip_pairwise:
        pairwise_dir.mkdir(parents=True, exist_ok=True)
        if args.pairwise_cmd:
            cmd = fill_template_command(args.pairwise_cmd, values)
        else:
            cmd = build_pairwise_cmd(
                pairwise_script=pairwise_script,
                root=root,
                patient_dir=patient_dir,
                ids_csv=simulator_available_csv,
                timeseries_csv=subset_csv,
                pairwise_dir=pairwise_dir,
                k_steps=args.k_steps,
                skip_mds=args.skip_mds,
                skip_plots=args.skip_plots,
                device=args.device,
                global_std_path=global_std_path,
                extra_args=args.pairwise_extra_args,
            )
        run_cmd(cmd, dry_run=args.dry_run)
    else:
        print(f"[skip] pairwise step; using {pairwise_dir}")

    tight_script = resolve_script(
        "tight_groups_script",
        args.tight_groups_script,
        [
            here / "find_tight_groups.py",
            root / "scripts" / "find_tight_groups.py",
            root / "data" / "find_tight_groups.py",
            root / "find_tight_groups.py",
        ],
    )

    if not args.skip_tight_groups:
        tight_groups_dir.mkdir(parents=True, exist_ok=True)
        if args.tight_groups_cmd:
            cmd = fill_template_command(args.tight_groups_cmd, values)
        else:
            cmd = [
                sys.executable, str(tight_script),
                "--in-dir", str(pairwise_dir),
                "--out-dir", str(tight_groups_dir),
                "--group-size", str(args.group_size),
            ]
        run_cmd(cmd, dry_run=args.dry_run)
    else:
        print(f"[skip] tight-groups step; using {tight_groups_dir}")

    if args.dry_run:
        print("[dry-run] stopping before parsing generated tight-group files.")
        return

    selected_ids, per_file = collect_tight_group_pids(tight_groups_dir, allowed=set(usable_ids))
    if not selected_ids:
        raise SystemExit(
            "No patient ids were parsed from tight-groups outputs. "
            f"Check files under {tight_groups_dir}."
        )

    write_pid_csv(selected_ids, output)

    summary = {
        "eligible_pids": str(eligible_pids),
        "csv": str(csv_path) if csv_ok else None,
        "patient_dir": str(patient_dir),
        "simulator_available_csv": str(simulator_available_csv),
        "subset_timeseries_csv": str(subset_csv) if subset_csv is not None else None,
        "pairwise_dir": str(pairwise_dir),
        "tight_groups_dir": str(tight_groups_dir),
        "group_size": int(args.group_size),
        "k_steps": int(args.k_steps),
        "skip_mds": bool(args.skip_mds),
        "skip_plots": bool(args.skip_plots),
        "global_std_path": str(global_std_path) if global_std_path else None,
        "output": str(output),
        "n_eligible": len(eligible_ids),
        "n_with_individual_pinn": len(usable_ids),
        "n_missing_individual_pinn": len(missing_ids),
        "n_selected_population": len(selected_ids),
        "selected_pids": sorted(selected_ids),
        "parsed_group_files": per_file,
    }
    summary_path = work_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[write] {summary_path}")

    print("\nDone.")
    print(f"Simulator-available CSV: {simulator_available_csv}")
    print(f"Population CSV: {output}")
    print(f"Selected patients: {len(selected_ids)}")


if __name__ == "__main__":
    main()
