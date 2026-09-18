"""Single-patient PINN training.

Trains the Physics-Informed Neural Network (PINN) simulator on a single
patient's MIMIC time-series.  This is the *first* of three PINN entry
points provided by this benchmark library:

  (1) scripts/train_pinn.py             (1 patient)
  (2) scripts/sweep_pinn.sh             (loop over many patients)
  (3) scripts/train_pinn_population.py  (one shared PINN over many patients)

Outputs (under ``--save_dir/<patient_<id>>/``)
----------------------------------------------
    pinn.pt
    medical_ode.pt
    neural_ode.pt
    pinn_fit.png                  observed vs PINN prediction
    pinn_fit_comparison.png       MedicalODE / NeuralODE / PINN side-by-side
    scales.npy                    {mean, std, ascl, state_min, state_max}
    init_state_norm.npy           first ICU timestep state (PINN norm space)
    _meta.json                    CLI args, csv path, git rev, timestamps

Usage
-----
    python scripts/train_pinn.py --patient_id 200325
    python scripts/train_pinn.py --patient_id 200325 --epochs 5000 --patience 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from data.config import device, PINN_IND_ROOT, state_dim
from data.loader import load_patients
from train.pinn.io_utils import (
    PINN_PHYSICAL_MAX,
    PINN_PHYSICAL_MIN,
    enforce_pinn_physical_bounds,
    git_rev,
    save_json,
    save_scales,
)
from train.pinn.train_pinn import train_pinn
from train.pinn.fast_pinn_loss import apply_fast_pinn_loss
from utils.visualization import plot_pinn_fit, plot_pinn_fit2


def _parse():
    p = argparse.ArgumentParser(
        description="Train a PINN simulator on a single patient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", type=int, required=True,
                   help="ICU stay ID to train on.")
    p.add_argument("--csv",        type=str,
                   default=str(_HERE.parent / "data" / "mimic_pinn_v4_filtered.csv"),
                   help="Patient time-series CSV.")
    p.add_argument("--save_dir",   type=str, default=PINN_IND_ROOT,
                   help="Output root directory; patient_<id>/ created inside.")
    p.add_argument("--epochs",     type=int,   default=10_000)
    p.add_argument("--patience",   type=int,   default=800)
    p.add_argument("--lr_pinn",    type=float, default=1e-4)
    p.add_argument("--lr_ode",     type=float, default=5e-5)
    p.add_argument("--no_fit_plots", action="store_true",
                   help="Skip writing pinn_fit*.png.")
    p.add_argument("--fast_loss", action="store_true",
                   help="Use vectorized rollout loss (O(R) instead of O(R^2)). "
                        "Numerically near-identical, but ~2x faster.  Recommended "
                        "for sweep / population training.")
    p.add_argument("--rollout_n_starts", type=int, default=None,
                   help="Only used with --fast_loss: random sub-sampling of "
                        "rollout starts per epoch (None = all starts).")
    return p.parse_args()


def main():
    args = _parse()

    out_dir = Path(args.save_dir) / f"patient_{args.patient_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  train_pinn  (single patient)")
    print(f"  patient_id : {args.patient_id}")
    print(f"  csv        : {args.csv}")
    print(f"  out_dir    : {out_dir}")
    print(f"  epochs     : {args.epochs}  patience={args.patience}")
    print(f"  lr_pinn    : {args.lr_pinn}  lr_ode={args.lr_ode}")
    print(f"  device     : {device}")
    print(f"  fast_loss  : {args.fast_loss}"
          + (f"  rollout_n_starts={args.rollout_n_starts}"
             if args.fast_loss else ""))
    print(f"{'='*60}\n")

    if args.fast_loss:
        apply_fast_pinn_loss(rollout_n_starts=args.rollout_n_starts)

    # ── Load patient + scales ────────────────────────────────────────────
    patient, _, scales = load_patients(args.csv, args.patient_id)
    scales = enforce_pinn_physical_bounds(scales)

    # ── Save scales.npy ──────────────────────────────────────────────────
    scales_path = out_dir / "scales.npy"
    save_scales(scales, scales_path)
    print(f"[Scales] Saved → {scales_path}")

    # Normalized initial state (same space as scales.npy) for downstream RL/eval.
    sc_np = np.load(str(scales_path), allow_pickle=True).item()
    x0 = patient["data"][0, 1 : 1 + state_dim].detach().cpu().numpy().astype(np.float32)
    x0 = np.clip(
        x0,
        sc_np["state_min"].astype(np.float32),
        sc_np["state_max"].astype(np.float32),
    ).astype(np.float32)
    init_path = out_dir / "init_state_norm.npy"
    np.save(str(init_path), x0)
    print(f"[Init]   Saved → {init_path}")

    # ── Train PINN ───────────────────────────────────────────────────────
    t0 = time.time()
    model, med_ode, neu_ode, best_loss = train_pinn(
        patient, scales,
        epochs   = args.epochs,
        patience = args.patience,
        lr_pinn  = args.lr_pinn,
        lr_ode   = args.lr_ode,
        save_dir = str(out_dir),
    )
    elapsed = time.time() - t0
    print(f"[PINN] Done. best_loss={best_loss.item():.6f}  "
          f"elapsed={elapsed/60:.1f} min")

    # ── Fitting figures ──────────────────────────────────────────────────
    if not args.no_fit_plots:
        try:
            plot_pinn_fit(model, patient, scales,
                          save_path=str(out_dir / "pinn_fit.png"))
            print(f"[PINN] Fitting figure → {out_dir / 'pinn_fit.png'}")
        except Exception as exc:
            print(f"[PINN] WARN  plot_pinn_fit failed: {exc}")
        try:
            plot_pinn_fit2(model, patient, scales,
                           save_path=str(out_dir / "pinn_fit_comparison.png"))
            print(f"[PINN] Comparison figure → {out_dir / 'pinn_fit_comparison.png'}")
        except Exception as exc:
            print(f"[PINN] WARN  plot_pinn_fit2 failed: {exc}")

    # ── _meta.json ───────────────────────────────────────────────────────
    meta = {
        "script":       "scripts/train_pinn.py",
        "patient_id":   int(args.patient_id),
        "csv":          str(Path(args.csv).resolve()),
        "save_dir":     str(out_dir.resolve()),
        "epochs":       int(args.epochs),
        "patience":     int(args.patience),
        "lr_pinn":      float(args.lr_pinn),
        "lr_ode":       float(args.lr_ode),
        "best_loss":    float(best_loss.item()),
        "elapsed_min":  round(elapsed / 60, 2),
        "git_rev":      git_rev(_HERE.parent),
        "argv":         sys.argv,
        "physical_min": PINN_PHYSICAL_MIN.tolist(),
        "physical_max": PINN_PHYSICAL_MAX.tolist(),
        "fast_loss":    bool(args.fast_loss),
        "rollout_n_starts": args.rollout_n_starts,
    }
    save_json(meta, out_dir / "_meta.json")
    print(f"[meta] Saved → {out_dir / '_meta.json'}")


if __name__ == "__main__":
    main()
