"""Population (shared) PINN training.

Trains ONE PINN simulator from MANY patients pooled together, using global
(across-patient) normalization scales.  This is the *third* of three PINN
entry points in this benchmark library:

  (1) scripts/train_pinn.py             (1 patient)
  (2) scripts/sweep_pinn.sh             (loop over many patients)
  (3) scripts/train_pinn_population.py  (one shared PINN)

Outputs (under ``--save_dir/``)
-------------------------------
    pinn.pt
    medical_ode.pt
    neural_ode.pt
    pinn_loss_history.png
    patient_fit_plots/<pid>_fit.png    (a few sampled patients)
    scales.npy                         {mean, std, ascl, state_min, state_max}
    patient_ids.txt                    list of patient IDs included
    _meta.json                         CLI args, csv path, git rev, timestamps

Usage
-----
    # Default: ids from checkpoints-cohort/cohort_1/cohort_1_training.csv (column pid / icu_id / …), time-series from --csv
    python scripts/train_pinn_population.py

    # All ICU ids in the time-series CSV (disables the default id list file)
    python scripts/train_pinn_population.py --patient_ids_csv ""

    # Specific patient list (overrides --patient_ids_csv)
    python scripts/train_pinn_population.py \
        --patient_ids 200325 201046 201101 201201 201299 \
        --save_dir results/pinn_population_5pat

    # Custom cohort file
    python scripts/train_pinn_population.py --patient_ids_csv path/to/ids.csv

    # Limit to first N patients (smoke test)
    python scripts/train_pinn_population.py --max_patients 10
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from data.config import device, PINN_POP_ROOT
from data.loader import load_all_patients
from train.pinn.io_utils import git_rev, save_json, save_scales
from train.pinn.train_cluster_pinn import train_cluster_pinn
from train.pinn.fast_pinn_loss import apply_fast_pinn_loss
from utils.visualization import plot_pinn_fit


# Physical bounds (real scale)
# Order: [SpO2, PaO2, Bilirubin, GCS, Urine_Step, Lactate]
# Matches train_allpatient_pinn.compute_scales:
#   only SpO2 (<=100) and GCS (3..15) get a finite upper bound.
PHYSICAL_MIN = np.array([0.,   0.,     0.,     3.,  0.,     0.    ], dtype=np.float32)
PHYSICAL_MAX = np.array([100., np.inf, np.inf, 15., np.inf, np.inf], dtype=np.float32)

DEFAULT_POPULATION_IDS_CSV = _HERE.parent / "data" / "population.csv"


def _enforce_physical_bounds(scales: dict) -> dict:
    """Intersect data-derived state_min/max with physical bounds.

    Mirrors ``train_allpatient_pinn.compute_scales`` (uses ``np.where`` to
    safely propagate inf entries of ``PHYSICAL_MAX``).
    """
    mean = scales["state_mean"].cpu().numpy().astype(np.float32)
    std  = scales["state_std" ].cpu().numpy().astype(np.float32)

    phys_min_norm = (PHYSICAL_MIN - mean) / std
    phys_max_norm = np.where(
        np.isfinite(PHYSICAL_MAX),
        (PHYSICAL_MAX - mean) / std,
        np.inf,
    ).astype(np.float32)

    phys_min_t = torch.tensor(phys_min_norm.astype(np.float32),
                              dtype=torch.float32, device=device)
    phys_max_t = torch.tensor(phys_max_norm,
                              dtype=torch.float32, device=device)

    sc = dict(scales)
    sc["state_min"] = torch.maximum(scales["state_min"], phys_min_t)
    sc["state_max"] = torch.minimum(scales["state_max"], phys_max_t)
    return sc


def _resolve_patient_ids(args) -> list[int] | None:
    """Return the list of patient ICU IDs to include, or None for all rows in ``--csv``."""
    if args.patient_ids:
        return [int(x) for x in args.patient_ids]
    csv_path = args.patient_ids_csv
    if csv_path is None or not str(csv_path).strip():
        return None
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise SystemExit(
            f"Patient id list CSV not found: {path}\n"
            "Create it, pass a valid --patient_ids_csv, use --patient_ids …, "
            "or pass --patient_ids_csv \"\" to use every icu_id in --csv."
        )
    df = pd.read_csv(path, comment="#")
    # Accept any common patient-id column name; fall back to first column.
    for candidate in ("icu_id", "pid", "patient_id", "stay_id"):
        if candidate in df.columns:
            col = candidate
            break
    else:
        col = df.columns[0]
    ids = sorted({int(x) for x in df[col].dropna().tolist()})
    print(f"[ids] read {len(ids)} patient ids from {path} (column '{col}')")
    return ids


def _save_loss_history(loss_history, out_path: Path) -> None:
    """Save loss curve.  Accepts either a list (legacy) or a dict of lists."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if isinstance(loss_history, dict):
            total = loss_history.get("total", [])
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].semilogy(total, color="#1f77b4", lw=1.5)
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
            axes[0].set_title("Population PINN training loss (total)")
            axes[0].grid(alpha=0.3)
            for key, color in [("data", "steelblue"), ("roll", "darkorange"),
                               ("ode", "green"), ("smooth", "gray")]:
                ys = loss_history.get(key, [])
                if ys:
                    axes[1].semilogy(ys, "-", color=color, lw=1.3, alpha=0.85, label=key)
            axes[1].set_title("Loss components"); axes[1].set_xlabel("Epoch")
            axes[1].legend(); axes[1].grid(alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(loss_history, color="#1f77b4", lw=1.5)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title("Population PINN training loss")
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"[Loss] Saved → {out_path}")
    except Exception as exc:
        print(f"[Loss] WARN  could not write {out_path}: {exc}")


def _save_patient_fit_samples(model, patients, scales, out_dir: Path,
                              n_sample: int = 5, seed: int = 42) -> None:
    if not patients:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sample = rng.sample(patients, min(n_sample, len(patients)))
    for p in sample:
        pid = int(p["icu_id"])
        try:
            plot_pinn_fit(model, p, scales,
                          save_path=str(out_dir / f"{pid}_fit.png"))
        except Exception as exc:
            print(f"[fit] WARN  patient {pid}: {exc}")


def _parse():
    p = argparse.ArgumentParser(
        description="Train ONE shared PINN simulator over many patients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=str,
                   default=str(_HERE.parent / "data" / "mimic_pinn_v4_filtered.csv"),
                   help="Patient time-series CSV.")
    p.add_argument("--save_dir", type=str,
                   default=PINN_POP_ROOT,
                   help="Output directory for the population PINN.")
    p.add_argument(
        "--patient_ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit ICU ids (overrides --patient_ids_csv).",
    )
    p.add_argument(
        "--patient_ids_csv",
        type=str,
        default=str(DEFAULT_POPULATION_IDS_CSV),
        help="CSV with cohort ids (column icu_id, pid, patient_id, stay_id, or first column). "
        f"Default: {DEFAULT_POPULATION_IDS_CSV}. Pass \"\" to use every icu_id in --csv.",
    )
    p.add_argument("--max_patients", type=int, default=None,
                   help="Cap number of patients (None = all).")

    # ── Hyperparameters (defaults match run_allpatient_pipeline.sh) ──────
    # epochs / patience overridden by the pipeline shell to 1000 / 200,
    # other LR / rollout / log values use the train_allpatient_pinn.py
    # script defaults (which the shell does not override).
    p.add_argument("--epochs",          type=int,   default=1000)
    p.add_argument("--patience",        type=int,   default=200)
    p.add_argument("--lr_pinn",         type=float, default=1e-4)
    p.add_argument("--lr_med",          type=float, default=5e-5,
                   help="MedicalODE learning rate.")
    p.add_argument("--lr_neural",       type=float, default=1e-4,
                   help="NeuralODE learning rate.")
    p.add_argument("--lr_ode",          type=float, default=None,
                   help="(legacy) shortcut that overrides both --lr_med and --lr_neural.")
    p.add_argument("--rollout_steps",   type=int,   default=20)
    p.add_argument("--n_per_epoch",     type=int,   default=0,
                   help="Patients sampled per epoch (0 = all).")
    p.add_argument("--patients_per_ep", type=int,   default=None,
                   help="(legacy) alias for --n_per_epoch.")
    p.add_argument("--log_interval",    type=int,   default=100)
    p.add_argument("--n_fit_samples",   type=int,   default=5,
                   help="How many patients to render in patient_fit_plots/.")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--per_patient_neural", action="store_true", default=True,
                   help="Build one PatientNeuralODE per patient "
                        "(64-64 Tanh, clamp +/-1) like train_allpatient_pinn.py. "
                        "Default: True.")
    p.add_argument("--shared_neural", dest="per_patient_neural",
                   action="store_false",
                   help="Use a single shared NeuralODE instead of per-patient.")
    p.add_argument("--no_fast_loss", action="store_true",
                   help="Disable the vectorized rollout loss (default: enabled "
                        "for population training).")
    p.add_argument("--rollout_n_starts", type=int, default=None,
                   help="Random sub-sampling of rollout starts per epoch "
                        "(only used when fast loss is enabled).")
    return p.parse_args()


def main():
    args = _parse()

    # ── Resolve legacy aliases ───────────────────────────────────────────
    if args.lr_ode is not None:
        args.lr_med    = args.lr_ode
        args.lr_neural = args.lr_ode
    if args.patients_per_ep is not None:
        args.n_per_epoch = int(args.patients_per_ep)

    # Multi-patient training: enable the vectorized rollout loss by default.
    # Has to happen BEFORE train_cluster_pinn is invoked, since that module
    # imports pinn_loss into its globals at call time.
    if not args.no_fast_loss:
        apply_fast_pinn_loss(rollout_n_starts=args.rollout_n_starts)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve requested patient IDs first, so global scales are pooled
    # ── ONLY over the patients we will actually train on (matches
    # ── train_allpatient_pinn.compute_scales).
    requested_ids = _resolve_patient_ids(args)

    print(f"[Load] reading {args.csv}")
    bundle = load_all_patients(
        args.csv,
        max_patients=args.max_patients,
        ids=requested_ids,
        state_bounds="p5p95",   # match train_allpatient_pinn.compute_scales
        margin=0.5,
    )
    patients = bundle.patients
    scales   = bundle.scales

    if requested_ids is not None:
        print(f"[Load] kept {len(patients)} / {len(requested_ids)} requested patients"
              f"  (scales pooled over these {len(patients)} patients)")
    else:
        print(f"[Load] using all {len(patients)} patients with data"
              f"  (scales pooled over these {len(patients)} patients)")

    if len(patients) < 2:
        raise SystemExit(f"Need at least 2 patients, got {len(patients)}.")

    # Apply physical bounds to scales (intersect with PHYSICAL_MIN/MAX)
    scales = _enforce_physical_bounds(scales)

    print(f"\n{'='*60}")
    print(f"  train_pinn_population  (one shared PINN over many patients)")
    print(f"  csv                : {args.csv}")
    print(f"  out_dir            : {out_dir}")
    print(f"  patients           : {len(patients)}")
    print(f"  epochs             : {args.epochs}  patience={args.patience}")
    print(f"  lr_pinn            : {args.lr_pinn}  lr_med={args.lr_med}  lr_neural={args.lr_neural}")
    print(f"  rollout_steps      : {args.rollout_steps}")
    print(f"  n_per_epoch        : {args.n_per_epoch}  (0 = all)")
    print(f"  per_patient_neural : {args.per_patient_neural}")
    print(f"  device             : {device}")
    print(f"{'='*60}\n")

    # ── Save scales.npy ──────────────────────────────────────────────────
    scales_path = out_dir / "scales.npy"
    save_scales(scales, scales_path)
    print(f"[Scales] Saved → {scales_path}")

    # ── Save patient_ids.txt ─────────────────────────────────────────────
    ids_path = out_dir / "patient_ids.txt"
    with open(ids_path, "w") as f:
        f.write("\n".join(str(int(p["icu_id"])) for p in patients) + "\n")
    print(f"[ids] Saved {len(patients)} ids → {ids_path}")

    # ── Train shared PINN ────────────────────────────────────────────────
    t0 = time.time()
    model, med_ode, neu_ode, best_loss, loss_history = train_cluster_pinn(
        cluster_patients   = patients,
        scales             = scales,
        epochs             = args.epochs,
        patience           = args.patience,
        lr_pinn            = args.lr_pinn,
        lr_med             = args.lr_med,
        lr_neural          = args.lr_neural,
        save_dir           = str(out_dir),
        rollout_steps      = args.rollout_steps,
        n_per_epoch        = args.n_per_epoch,
        log_interval       = args.log_interval,
        seed               = args.seed,
        per_patient_neural = args.per_patient_neural,
    )
    elapsed = time.time() - t0
    print(f"[PINN] Done. best_loss={best_loss.item():.6f}  "
          f"elapsed={elapsed/60:.1f} min")

    # ── Loss history JSON + curve + sample fits ──────────────────────────
    try:
        save_json(loss_history, out_dir / "loss_history.json")
        print(f"[Loss] history JSON → {out_dir / 'loss_history.json'}")
    except Exception as exc:
        print(f"[Loss] WARN  could not write loss_history.json: {exc}")

    _save_loss_history(loss_history, out_dir / "pinn_loss_history.png")
    _save_patient_fit_samples(model, patients, scales,
                              out_dir / "patient_fit_plots",
                              n_sample=args.n_fit_samples)

    # ── _meta.json ───────────────────────────────────────────────────────
    meta = {
        "script":          "scripts/train_pinn_population.py",
        "csv":             str(Path(args.csv).resolve()),
        "save_dir":        str(out_dir.resolve()),
        "n_patients":      len(patients),
        "patient_ids_src": (
            "explicit"
            if args.patient_ids
            else (
                "all_mimic_csv"
                if not str(args.patient_ids_csv or "").strip()
                else "csv"
            )
        ),
        "patient_ids_csv": str(Path(args.patient_ids_csv).resolve())
        if str(args.patient_ids_csv or "").strip()
        else None,
        "epochs":          int(args.epochs),
        "patience":        int(args.patience),
        "lr_pinn":         float(args.lr_pinn),
        "lr_med":          float(args.lr_med),
        "lr_neural":       float(args.lr_neural),
        "rollout_steps":      int(args.rollout_steps),
        "n_per_epoch":        int(args.n_per_epoch),
        "log_interval":       int(args.log_interval),
        "seed":               int(args.seed),
        "per_patient_neural": bool(args.per_patient_neural),
        "best_loss":       float(best_loss.item()),
        "elapsed_min":     round(elapsed / 60, 2),
        "git_rev":         git_rev(_HERE.parent),
        "argv":            sys.argv,
        "physical_min":    PHYSICAL_MIN.tolist(),
        "physical_max":    PHYSICAL_MAX.tolist(),
        "fast_loss":       not bool(args.no_fast_loss),
        "rollout_n_starts": args.rollout_n_starts,
    }
    save_json(meta, out_dir / "_meta.json")
    print(f"[meta] Saved → {out_dir / '_meta.json'}")


if __name__ == "__main__":
    main()
