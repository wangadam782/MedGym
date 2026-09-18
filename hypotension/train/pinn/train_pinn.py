"""PINN training for a single patient (or a batch of patients).

Loss = data term + w_phys * physics term + w_roll * multi-step rollout term
       + w_resid * residual regularisation + w_range * physiological range term

  Data term     computed on observation mask only. Mask from (M) column:
                real measurements weight 1.0, GAN-imputed weight w_imputed.
  Physics term  PINN instantaneous derivative should match MedicalODE + NeuralODE.
  Rollout term  multi-step prediction error. Critical for PINN to work as simulator.
  Residual reg  suppresses NeuralODE, keeping physics term dominant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hypotension.data.config import SIDX, device, state_dim
from hypotension.data.loader import split_data
from hypotension.models.medical_ode import MedicalODE, MedicalODENorm, NeuralODE, PatientNeuralODE
from hypotension.models.pinn import PINN
from hypotension.train.pinn.io_utils import norm_bounds


# =============================================================================
def make_windows(T: int, K: int):
    """All rollout starting points for windows of length K."""
    return np.arange(0, max(1, T - K))


def pinn_loss(pinn, med_n, neu, pat, x, u, m, lo, hi, dt,
              w_phys=0.3, w_roll=1.0, w_resid=1e-3, w_range=1e-2,
              roll_K=5, n_substeps=8, roll_starts=None):
    T = x.shape[0]
    out = {}

    # ---- Data term: one-step, masked ----
    xp = pinn.step(x[:-1], u[:-1], dt, lo, hi, n_substeps)
    mm = m[1:]
    data = (((xp - x[1:]) ** 2) * mm).sum() / mm.sum().clamp(min=1)
    out["data"] = float(data.detach())

    # ---- Physics term ----
    f_phys = med_n(x, u) + neu(x, u) + (pat(x, u) if pat is not None else 0.0)
    phys = F.mse_loss(pinn(x, u), f_phys)
    out["phys"] = float(phys.detach())

    # ---- Multi-step rollout term ----
    roll = torch.zeros((), device=x.device)
    if w_roll > 0 and T > roll_K + 1:
        starts = make_windows(T, roll_K) if roll_starts is None else roll_starts
        xs = x[starts]
        for j in range(roll_K):
            xs = pinn.step(xs, u[starts + j], dt, lo, hi, n_substeps)
            tgt, msk = x[starts + j + 1], m[starts + j + 1]
            roll = roll + (((xs - tgt) ** 2) * msk).sum() / msk.sum().clamp(min=1)
        roll = roll / roll_K
    out["roll"] = float(roll.detach())

    resid = (neu(x, u) ** 2).mean()
    rng = (torch.relu(lo - xp) ** 2 + torch.relu(xp - hi) ** 2).mean()
    out["resid"], out["range"] = float(resid.detach()), float(rng.detach())

    total = data + w_phys * phys + w_roll * roll + w_resid * resid + w_range * rng
    out["total"] = float(total.detach())
    return total, out


# =============================================================================
def train_pinn(patient, scales, epochs=4000, patience=600,
               lr_pinn=1e-3, lr_ode=5e-4, save_dir=None,
               w_phys=0.3, w_roll=1.0, roll_K=5, n_substeps=8,
               use_patient_ode=True, no_physics=False, seed=0, verbose=True):
    """Train a PINN for one patient. Returns (pinn, medical_ode, neural_ode, best_loss)."""
    torch.manual_seed(seed)
    _, x_all, u_all = split_data(patient)
    m_all = patient["mask"]
    n_tr = patient.get("n_train", len(x_all))
    x, u, m = x_all[:n_tr], u_all[:n_tr], m_all[:n_tr]
    if verbose:
        print(f"  Temporal split: train t=0..{n_tr-1}  out-of-sample t={n_tr}..{len(x_all)-1}")
    lo_np, hi_np = norm_bounds(scales)
    lo = torch.as_tensor(lo_np, device=device)
    hi = torch.as_tensor(hi_np, device=device)
    dt = float(scales.get("dt", 1.0))

    pinn = PINN().to(device)
    core = MedicalODE().to(device)
    med_n = MedicalODENorm(core, scales).to(device)
    neu = NeuralODE().to(device)
    pat = PatientNeuralODE().to(device) if use_patient_ode else None

    groups = [{"params": pinn.parameters(), "lr": lr_pinn},
              {"params": neu.parameters(), "lr": lr_pinn}]
    if pat is not None:
        groups.append({"params": pat.parameters(), "lr": lr_pinn})
    if not no_physics:
        groups.append({"params": core.parameters(), "lr": lr_ode})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    wp = 0.0 if no_physics else w_phys
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        opt.zero_grad()
        loss, log = pinn_loss(pinn, med_n, neu, pat, x, u, m, lo, hi, dt,
                              w_phys=wp, w_roll=w_roll, roll_K=roll_K,
                              n_substeps=n_substeps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], 1.0)
        opt.step(); sched.step()

        monitor = log["data"] + log["roll"]
        if monitor < best - 1e-6:
            best, bad = monitor, 0
            best_state = {k: {kk: vv.detach().clone() for kk, vv in mod.state_dict().items()}
                          for k, mod in [("pinn", pinn), ("core", core),
                                         ("neu", neu)] + ([("pat", pat)] if pat else [])}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop @ ep {ep} (patience={patience})")
                break
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  ep {ep:5d}  total {log['total']:8.4f}  data {log['data']:8.4f}"
                  f"  roll {log['roll']:8.4f}  phys {log['phys']:7.4f}"
                  f"  resid {log['resid']:6.3f}")

    if best_state:
        pinn.load_state_dict(best_state["pinn"]); core.load_state_dict(best_state["core"])
        neu.load_state_dict(best_state["neu"])
        if pat is not None:
            pat.load_state_dict(best_state["pat"])

    if save_dir:
        d = Path(save_dir); d.mkdir(parents=True, exist_ok=True)
        torch.save(pinn.state_dict(), d / "pinn.pt")
        torch.save(core.state_dict(), d / "medical_ode.pt")
        torch.save(neu.state_dict(), d / "neural_ode.pt")
        if pat is not None:
            torch.save(pat.state_dict(), d / "patient_ode.pt")

    return pinn, core, neu, torch.tensor(best)


def load_trained(save_dir, scales):
    """Load trained model; hidden size inferred automatically from checkpoint."""
    d = Path(save_dir)
    sd = torch.load(d / "pinn.pt", map_location=device)
    hid = sd["net.0.weight"].shape[0]
    pinn = PINN(hidden=hid).to(device); pinn.load_state_dict(sd)
    core = MedicalODE().to(device); core.load_state_dict(
        torch.load(d / "medical_ode.pt", map_location=device))
    nsd = torch.load(d / "neural_ode.pt", map_location=device)
    neu = NeuralODE(hidden=nsd["net.0.weight"].shape[0]).to(device); neu.load_state_dict(nsd)
    pat = None
    if (d / "patient_ode.pt").exists():
        pat = PatientNeuralODE().to(device)
        pat.load_state_dict(torch.load(d / "patient_ode.pt", map_location=device))
    return pinn.eval(), core.eval(), neu.eval(), pat
