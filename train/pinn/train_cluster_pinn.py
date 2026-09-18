"""train/pinn/train_cluster_pinn.py — Population PINN training.

Trains one shared PINN (+ MedicalODE + NeuralODE) over many patients using
global normalization scales.

The training loop, best-model selection criterion, and default
hyperparameters are aligned with the allpatient variant:

  * 1 epoch = sweep over all patients (or a random subsample of size
    ``n_per_epoch``).  Each patient triggers a forward/backward/step.
  * Three Adam optimizers — PINN / MedicalODE / NeuralODE — with separate
    learning rates and CosineAnnealingLR schedulers stepping once per
    epoch.
  * Per-patient ``dx/dt`` clipping ranges (computed from each patient's
    p1–p99 of the observed derivatives).
  * Per-optimizer gradient clipping (``max_norm=1.0``).
  * Best model is judged by **epoch-averaged** total loss.
  * Optional **per-patient NeuralODE** (``per_patient_neural=True``):
    one lightweight ``PatientNeuralODE`` (64-64 Tanh, clamp ±1) is built
    per patient and saved as ``neural_ode_pid{pid}.pt``.

Returns
-------
(model, medical_ode, neural_ode_or_dict, best_loss_tensor, loss_history_dict)
    * ``neural_ode_or_dict``:
        - ``per_patient_neural=False`` (default): single ``NeuralODE``
        - ``per_patient_neural=True``:  ``dict[icu_id -> PatientNeuralODE]``
    * ``loss_history_dict`` keys:
      ``{"total", "data", "roll", "ode", "smooth"}``.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.optim as optim

from data.config import device, state_dim, action_dim
from models import PINN, MedicalODE, NeuralODE, PatientNeuralODE
from .train_pinn import compute_dxdt_clip_vals, pinn_loss


def train_cluster_pinn(
    cluster_patients:   list[dict],
    scales:             dict,
    epochs:             int   = 5000,
    patience:           int   = 600,
    lr_pinn:            float = 1e-4,
    lr_med:             float = 5e-5,
    lr_neural:          float = 1e-4,
    save_dir:           str   = "results",
    rollout_steps:      int   = 20,
    n_per_epoch:        int   = 0,
    log_interval:       int   = 100,
    seed:               int   = 42,
    per_patient_neural: bool  = False,
    # ── backward-compatibility aliases ───────────────────────────────
    lr_ode:             float | None = None,
    patients_per_ep:    int   | None = None,
) -> tuple:
    """Train a shared PINN over many patients.

    Returns (model, medical_ode, neural_ode_or_dict, best_loss, loss_history).
    """
    n = len(cluster_patients)
    if n == 0:
        raise ValueError("cluster_patients is empty")

    # Backward-compat argument handling
    if lr_ode is not None:
        lr_med    = lr_ode
        lr_neural = lr_ode
    if patients_per_ep is not None:
        n_per_epoch = int(patients_per_ep)

    # ── Models ───────────────────────────────────────────────────────
    model       = PINN(state_dim, action_dim).to(device)
    medical_ode = MedicalODE().to(device)

    if per_patient_neural:
        pids = [int(p["icu_id"]) for p in cluster_patients]
        if len(set(pids)) != n:
            raise ValueError("cluster_patients contains duplicate icu_id values; "
                             "per_patient_neural=True requires unique ids.")
        neural_modules: dict[int, PatientNeuralODE] = {
            pid: PatientNeuralODE(state_dim, action_dim).to(device) for pid in pids
        }
        neural_params = [pp for node in neural_modules.values()
                              for pp in node.parameters()]
    else:
        neural_module = NeuralODE().to(device)
        neural_modules = None  # type: ignore[assignment]
        neural_params  = list(neural_module.parameters())

    pinn_optim   = optim.Adam(model.parameters(),       lr=lr_pinn,   weight_decay=1e-4)
    med_optim    = optim.Adam(medical_ode.parameters(), lr=lr_med,    weight_decay=1e-3)
    neural_optim = optim.Adam(neural_params,            lr=lr_neural, weight_decay=1e-4)

    pinn_sched   = optim.lr_scheduler.CosineAnnealingLR(pinn_optim,   T_max=epochs, eta_min=1e-6)
    med_sched    = optim.lr_scheduler.CosineAnnealingLR(med_optim,    T_max=epochs, eta_min=1e-7)
    neural_sched = optim.lr_scheduler.CosineAnnealingLR(neural_optim, T_max=epochs, eta_min=1e-6)

    # Per-patient dx/dt clip ranges (p1-p99 × 1.5 of observed derivatives)
    dxdt_clip_by_idx: list[list[float]] = []
    for p in cluster_patients:
        try:
            cv = compute_dxdt_clip_vals(p["data"], p["mask"], p["dt"])
        except Exception:
            cv = [5.0] * state_dim
        dxdt_clip_by_idx.append(cv)

    n_per_epoch = max(0, int(n_per_epoch))
    sweep_label = "all" if (n_per_epoch == 0 or n_per_epoch >= n) else str(n_per_epoch)
    print(f"\n[ClusterPINN] {n} patients  n_per_epoch={sweep_label}"
          f"  rollout_steps={rollout_steps}  per_patient_neural={per_patient_neural}")
    print(f"  lr_pinn={lr_pinn}  lr_med={lr_med}  lr_neural={lr_neural}")
    print(f"  epochs={epochs}  patience={patience}  log_interval={log_interval}")

    best_loss  = float("inf")
    best_state: dict | None = None
    no_improve = 0
    rng        = np.random.default_rng(seed)
    history: dict[str, list[float]] = {
        "total": [], "data": [], "roll": [], "ode": [], "smooth": [],
    }
    p2, p3 = epochs // 2, epochs * 2 // 3

    for ep in range(epochs):
        if ep < p2:
            w = dict(w_data=20., w_roll=40., w_ode=10., w_smooth=0.1)
        elif ep < p3:
            w = dict(w_data=10., w_roll=20., w_ode=20., w_smooth=0.05)
        else:
            w = dict(w_data=5.,  w_roll=20., w_ode=20., w_smooth=0.02)

        if 0 < n_per_epoch < n:
            idx_order = rng.choice(n, size=n_per_epoch, replace=False).tolist()
        else:
            idx_order = rng.permutation(n).tolist()

        ep_losses = {"total": 0., "data": 0., "roll": 0., "ode": 0., "smooth": 0.}
        n_done = 0

        for idx in idx_order:
            p = cluster_patients[idx]
            if per_patient_neural:
                neural_for_loss = neural_modules[int(p["icu_id"])]
            else:
                neural_for_loss = neural_module

            loss, parts = pinn_loss(
                model, scales, medical_ode, neural_for_loss,
                p["data"], p["mask"], p["dt"],
                dxdt_clip_by_idx[idx],
                **w,
                rollout_steps=rollout_steps,
                return_parts=True,
            )
            if torch.isnan(loss):
                continue

            pinn_optim.zero_grad()
            med_optim.zero_grad()
            neural_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),       max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(medical_ode.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(neural_for_loss.parameters(), max_norm=1.0)
            pinn_optim.step()
            med_optim.step()
            neural_optim.step()

            ep_losses["total"] += float(loss.detach())
            for k in ("data", "roll", "ode", "smooth"):
                ep_losses[k] += parts[k]
            n_done += 1

        if n_done == 0:
            print(f"ep{ep}: all patients produced NaN, skipping epoch.")
            continue

        for k in ep_losses:
            ep_losses[k] /= n_done
            history[k].append(ep_losses[k])

        pinn_sched.step()
        med_sched.step()
        neural_sched.step()

        # Best by epoch-averaged total loss
        if ep_losses["total"] < best_loss:
            best_loss  = ep_losses["total"]
            new_best = {
                "model":       {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "medical_ode": {k: v.detach().cpu().clone() for k, v in medical_ode.state_dict().items()},
            }
            if per_patient_neural:
                new_best["neural_odes"] = {
                    pid: {k: v.detach().cpu().clone() for k, v in node.state_dict().items()}
                    for pid, node in neural_modules.items()
                }
            else:
                new_best["neural_ode"] = {
                    k: v.detach().cpu().clone() for k, v in neural_module.state_dict().items()
                }
            best_state = new_best
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop ep={ep}  best={best_loss:.5f}")
                break

        if ep % log_interval == 0:
            sc  = torch.nn.functional.softplus(model.log_scale).detach().cpu().numpy()
            tag = "P1" if ep < p2 else "P2" if ep < p3 else "P3"
            print(f"  ep {ep:5d}[{tag}]  loss={ep_losses['total']:.4f}  "
                  f"data={ep_losses['data']:.3f}  roll={ep_losses['roll']:.3f}  "
                  f"ode={ep_losses['ode']:.3f}  scales={sc.round(3)}")

    # ── Restore best state ───────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        medical_ode.load_state_dict(best_state["medical_ode"])
        if per_patient_neural:
            for pid, node in neural_modules.items():
                node.load_state_dict(best_state["neural_odes"][pid])
        else:
            neural_module.load_state_dict(best_state["neural_ode"])

    # ── Save artifacts ───────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(),       f"{save_dir}/pinn.pt")
    torch.save(medical_ode.state_dict(), f"{save_dir}/medical_ode.pt")
    if per_patient_neural:
        for pid, node in neural_modules.items():
            torch.save(node.state_dict(), f"{save_dir}/neural_ode_pid{pid}.pt")
    else:
        torch.save(neural_module.state_dict(), f"{save_dir}/neural_ode.pt")

    np.savez(
        f"{save_dir}/scales.npz",
        state_mean=scales["state_mean"].cpu().numpy(),
        state_std=scales["state_std"].cpu().numpy(),
        state_min=scales["state_min"].cpu().numpy(),
        state_max=scales["state_max"].cpu().numpy(),
        action_scale=scales["action_scale"].cpu().numpy(),
    )
    print(f"\n[ClusterPINN] Saved to {save_dir}/  best_loss={best_loss:.6f}")

    neural_returned = neural_modules if per_patient_neural else neural_module
    return model, medical_ode, neural_returned, torch.tensor(best_loss), history
