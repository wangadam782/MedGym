"""Population PINN training for the hypotension benchmark.

Run from the medrl-tacos root:

    python scripts/hypotension/train_pinn_population.py --n_patients 40 --epochs 3000

A single model is trained over N patients; cluster-pooled PINNs use
--patient_ids to specify the cluster members explicitly.

Outputs (under --save_dir/):
    pinn.pt  medical_ode.pt  neural_ode.pt  scales.npy  _meta.json
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

from hypotension.data.config import PINN_POP_ROOT, PROCESSED_ROOT, device, state_dim
from hypotension.data.loader import list_patients, load_patients, split_data
from hypotension.models.medical_ode import MedicalODE, MedicalODENorm, NeuralODE
from hypotension.models.pinn import PINN
from hypotension.train.pinn.io_utils import (enforce_pinn_physical_bounds, git_rev,
                                              norm_bounds, save_json, save_scales)
from hypotension.train.pinn.train_pinn import pinn_loss


def _parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_patients", type=int, default=40)
    p.add_argument("--patient_ids", type=int, nargs="+",
                   default=[26, 7, 52, 59, 43, 32, 4, 57, 35, 48])
    p.add_argument("--processed", type=str, default=str(PROCESSED_ROOT))
    p.add_argument("--save_dir", type=str, default=str(PINN_POP_ROOT / "clu10"))
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=400)
    p.add_argument("--lr_pinn", type=float, default=1e-3)
    p.add_argument("--lr_ode", type=float, default=5e-4)
    p.add_argument("--w_phys", type=float, default=0.3)
    p.add_argument("--w_roll", type=float, default=1.0)
    p.add_argument("--roll_K", type=int, default=5)
    p.add_argument("--n_substeps", type=int, default=8)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--hidden", type=int, default=64,
                   help="Smaller hidden size for population model suppresses overfitting")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = _parse()
    torch.manual_seed(a.seed)
    out = Path(a.save_dir); out.mkdir(parents=True, exist_ok=True)

    ids = a.patient_ids if a.patient_ids is not None else list_patients(a.processed)[: a.n_patients]
    print(f"[pop] {len(ids)} patients, hidden={a.hidden}, train_frac={a.train_frac}")

    pats, scales = [], None
    for pid in ids:
        p, _, sc = load_patients(a.processed, pid, a.train_frac)
        scales = enforce_pinn_physical_bounds(sc)
        pats.append(p)
    save_scales(scales, out / "scales.npy")

    lo = torch.as_tensor(norm_bounds(scales)[0], device=device)
    hi = torch.as_tensor(norm_bounds(scales)[1], device=device)
    dt = float(scales.get("dt", 1.0))

    pinn = PINN(hidden=a.hidden).to(device)
    core = MedicalODE().to(device)
    med_n = MedicalODENorm(core, scales).to(device)
    neu = NeuralODE(hidden=a.hidden).to(device)
    opt = torch.optim.Adam([{"params": pinn.parameters(), "lr": a.lr_pinn},
                            {"params": neu.parameters(), "lr": a.lr_pinn},
                            {"params": core.parameters(), "lr": a.lr_ode}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    cache = []
    for p in pats:
        _, x, u = split_data(p)
        n = p["n_train"]
        cache.append((x[:n], u[:n], p["mask"][:n]))

    best, bad, best_state = float("inf"), 0, None
    t0 = time.time()
    for ep in range(a.epochs):
        opt.zero_grad(); tot = 0.0; logs = None
        for x, u, m in cache:
            l, lg = pinn_loss(pinn, med_n, neu, None, x, u, m, lo, hi, dt,
                              w_phys=a.w_phys, w_roll=a.w_roll, roll_K=a.roll_K,
                              n_substeps=a.n_substeps)
            (l / len(cache)).backward()
            tot += lg["data"] + lg["roll"]; logs = lg
        torch.nn.utils.clip_grad_norm_(
            list(pinn.parameters()) + list(neu.parameters()) + list(core.parameters()), 1.0)
        opt.step(); sched.step()
        tot /= len(cache)
        if tot < best - 1e-6:
            best, bad = tot, 0
            best_state = {k: {kk: vv.detach().clone() for kk, vv in md.state_dict().items()}
                          for k, md in [("pinn", pinn), ("core", core), ("neu", neu)]}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop @ ep {ep}"); break
        if ep % max(1, a.epochs // 10) == 0:
            print(f"  ep {ep:5d}  data+roll {tot:8.4f}  phys {logs['phys']:7.4f}")

    if best_state:
        pinn.load_state_dict(best_state["pinn"]); core.load_state_dict(best_state["core"])
        neu.load_state_dict(best_state["neu"])
    torch.save(pinn.state_dict(), out / "pinn.pt")
    torch.save(core.state_dict(), out / "medical_ode.pt")
    torch.save(neu.state_dict(), out / "neural_ode.pt")

    root = Path(__file__).resolve().parent.parent.parent
    save_json(dict(script="scripts/hypotension/train_pinn_population.py",
                   n_patients=len(ids), patient_ids=ids, hidden=a.hidden,
                   epochs=a.epochs, best_loss=best,
                   elapsed_min=round((time.time()-t0)/60, 2),
                   train_frac=a.train_frac, git_rev=git_rev(root),
                   argv=sys.argv), out / "_meta.json")
    print(f"[pop] done. best={best:.6f}  {(time.time()-t0)/60:.1f} min  -> {out}")


if __name__ == "__main__":
    main()
