"""Single-patient PINN training for the hypotension benchmark.

Run from the medrl-tacos root:

    python scripts/hypotension/train_pinn.py --patient_id 0
    python scripts/hypotension/train_pinn.py --patient_id 0 --epochs 6000 --patience 800

Outputs (under --save_dir/patient_<id>/):
    pinn.pt  medical_ode.pt  neural_ode.pt  patient_ode.pt
    scales.npy  init_state_norm.npy  _meta.json  learned_params.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

from hypotension.data.config import PINN_IND_ROOT, PROCESSED_ROOT, device, state_dim
from hypotension.data.loader import load_patients
from hypotension.train.pinn.io_utils import (PINN_PHYSICAL_MAX, PINN_PHYSICAL_MIN,
                                              enforce_pinn_physical_bounds, git_rev,
                                              save_json, save_scales)
from hypotension.train.pinn.train_pinn import train_pinn


def _parse():
    p = argparse.ArgumentParser(description="Train hypotension PINN on a single patient.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--patient_id", type=int, required=True)
    p.add_argument("--processed", type=str, default=str(PROCESSED_ROOT))
    p.add_argument("--save_dir",  type=str, default=str(PINN_IND_ROOT))
    p.add_argument("--epochs",    type=int,   default=4000)
    p.add_argument("--patience",  type=int,   default=600)
    p.add_argument("--lr_pinn",   type=float, default=1e-3)
    p.add_argument("--lr_ode",    type=float, default=5e-4)
    p.add_argument("--w_phys",    type=float, default=0.3)
    p.add_argument("--w_roll",    type=float, default=1.0)
    p.add_argument("--roll_K",    type=int,   default=5)
    p.add_argument("--n_substeps", type=int,  default=8)
    p.add_argument("--no_patient_ode", action="store_true")
    p.add_argument("--no_physics", action="store_true")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    a = _parse()
    out = Path(a.save_dir) / f"patient_{a.patient_id}"
    out.mkdir(parents=True, exist_ok=True)

    if not a.quiet:
        print(f"\n{'='*60}")
        print(f"  train_pinn  (hypotension, single patient)")
        print(f"  patient_id : {a.patient_id}")
        print(f"  processed  : {a.processed}")
        print(f"  out_dir    : {out}")
        print(f"  epochs     : {a.epochs}  patience={a.patience}")
        print(f"  device     : {device}")
        print(f"{'='*60}\n")

    patient, _, scales = load_patients(a.processed, a.patient_id, a.train_frac)
    scales = enforce_pinn_physical_bounds(scales)
    save_scales(scales, out / "scales.npy")

    x0 = patient["data"][0, 1:1 + state_dim].detach().cpu().numpy().astype(np.float32)
    np.save(str(out / "init_state_norm.npy"), x0)

    t0 = time.time()
    pinn, core, neu, best = train_pinn(
        patient, scales, epochs=a.epochs, patience=a.patience,
        lr_pinn=a.lr_pinn, lr_ode=a.lr_ode, save_dir=str(out),
        w_phys=a.w_phys, w_roll=a.w_roll, roll_K=a.roll_K,
        n_substeps=a.n_substeps, use_patient_ode=not a.no_patient_ode,
        no_physics=a.no_physics, seed=a.seed, verbose=not a.quiet)
    el = time.time() - t0
    print(f"[PINN] done. best(data+roll)={float(best):.6f}  {el/60:.2f} min")

    lp = core.learned_params()
    save_json({k: dict(learned=v, ref=getattr(core.r, k), ratio=v / getattr(core.r, k))
               for k, v in lp.items()}, out / "learned_params.json")

    root = Path(__file__).resolve().parent.parent.parent
    save_json(dict(script="scripts/hypotension/train_pinn.py",
                   patient_id=int(a.patient_id),
                   processed=str(Path(a.processed).resolve()),
                   save_dir=str(out.resolve()), epochs=a.epochs, patience=a.patience,
                   lr_pinn=a.lr_pinn, lr_ode=a.lr_ode, w_phys=a.w_phys,
                   train_frac=a.train_frac, w_roll=a.w_roll, roll_K=a.roll_K,
                   best_loss=float(best), elapsed_min=round(el / 60, 2),
                   git_rev=git_rev(root), argv=sys.argv,
                   physical_min=PINN_PHYSICAL_MIN.tolist(),
                   physical_max=PINN_PHYSICAL_MAX.tolist()), out / "_meta.json")
    print(f"[meta] -> {out/'_meta.json'}")


if __name__ == "__main__":
    main()
