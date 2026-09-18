"""PINN training: loss functions and training loop."""
import os
import numpy as np
import torch
import torch.optim as optim

from data.config import device, state_features, state_dim, action_dim
from models import PINN, MedicalODE, NeuralODE


def compute_dxdt_clip_vals(data, mask, dt, dt_thresh: float = 0.05):
    """Compute per-variable upper bounds for dx/dt clipping (p1-p99 of data)."""
    x_full = data[:, 1:]
    x      = x_full[:, :state_dim]
    mask_x = mask[:, 1:1+state_dim]

    vm     = dt >= dt_thresh
    x_t    = x[:-1][vm]
    x_next = x[1:][vm]
    mk_t   = mask_x[:-1][vm]
    mk_n   = mask_x[1:][vm]
    both   = mk_t * mk_n
    dt_v   = dt[vm].unsqueeze(1).clamp(min=dt_thresh)
    dxdt   = (x_next - x_t) / dt_v

    clip_vals = []
    for i in range(state_dim):
        obs = both[:, i] > 0.5
        if obs.sum() > 5:
            vals   = dxdt[obs, i].cpu().numpy()
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            margin = max(abs(lo), abs(hi)) * 1.5
            clip_vals.append(float(margin))
        else:
            clip_vals.append(5.0)

    return clip_vals


def pinn_loss(
    model, scales, medical_ode, neural_ode,
    data, mask, dt,
    dxdt_clip_vals,
    w_data: float = 20., w_roll: float = 20.,
    w_ode:  float = 0.,  w_smooth: float = 0.1,
    dt_noise_thresh: float = 0.05,
    rollout_steps:   int   = 30,
    return_parts:    bool  = False,
):
    x_full = data[:, 1:]
    x      = x_full[:, :state_dim]
    a      = x_full[:, state_dim:]
    mask_x = mask[:, 1:1+state_dim]
    T      = x.shape[0]

    x_t       = x[:-1]
    x_next    = x[1:]
    a_t       = a[:-1]
    mask_t    = mask_x[:-1]
    mask_next = mask_x[1:]
    dt_t      = dt.unsqueeze(1)

    dxdt_pred = model(x_t, a_t)

    valid_dt      = dt >= dt_noise_thresh
    both_observed = mask_t * mask_next
    valid_mask    = both_observed * valid_dt.float().unsqueeze(1)

    dt_safe     = dt_t.clamp(min=dt_noise_thresh)
    target_dxdt = (x_next - x_t) / dt_safe

    clip_t      = torch.tensor(dxdt_clip_vals, dtype=torch.float32, device=data.device)
    target_dxdt = torch.max(torch.min(target_dxdt, clip_t), -clip_t)

    data_loss = ((dxdt_pred - target_dxdt) ** 2 * valid_mask).sum() \
                / (valid_mask.sum() + 1e-6)

    roll_total, roll_count = torch.tensor(0., device=data.device), 0
    for k in range(2, rollout_steps + 1):
        n = T - k
        if n <= 0:
            break
        x_curr = x[:n].clone()
        for step in range(k):
            a_step  = a[step:n+step]
            dt_step = dt[step:n+step].unsqueeze(1)
            dx      = model(x_curr, a_step)
            x_curr  = x_curr + dx * dt_step
            x_curr  = torch.max(
                torch.min(x_curr, scales["state_max"]),
                scales["state_min"],
            )
        target_k = x[k:n+k]
        mask_k   = mask_x[:n] * mask_x[k:n+k]
        if target_k.shape[0] > 0:
            loss_k     = ((x_curr - target_k) ** 2 * mask_k).sum() \
                         / (mask_k.sum() + 1e-6)
            roll_total = roll_total + loss_k
            roll_count += 1

    roll_loss = roll_total / max(roll_count, 1)

    smooth_loss = (
        torch.mean((dxdt_pred[1:] - dxdt_pred[:-1]) ** 2)
        if dxdt_pred.shape[0] > 1
        else torch.tensor(0., device=data.device)
    )

    if w_ode == 0. or valid_dt.sum() <= 1:
        ode_loss = torch.tensor(0., device=data.device)
    else:
        x_v    = x_t[valid_dt]
        a_v    = a_t[valid_dt]
        mask_v = both_observed[valid_dt]
        dxdt_p = dxdt_pred[valid_dt]
        dxdt_o = medical_ode(x_v, a_v) + neural_ode(x_v, a_v)
        ode_res  = torch.clamp(dxdt_p - dxdt_o, -5., 5.)
        ode_loss = (ode_res ** 2 * mask_v).sum() / (mask_v.sum() + 1e-6)

    total = (w_data   * data_loss
             + w_roll   * roll_loss
             + w_smooth * smooth_loss
             + w_ode    * ode_loss)

    if return_parts:
        return total, {
            "data":   float(data_loss.detach()),
            "roll":   float(roll_loss.detach()),
            "ode":    float(ode_loss.detach()),
            "smooth": float(smooth_loss.detach()),
        }
    return total


def train_pinn(
    patient, scales,
    epochs:   int = 10000,
    patience: int = 800,
    lr_pinn:  float = 1e-4,
    lr_ode:   float = 5e-5,
    save_dir: str  = "results",
):
    model       = PINN(state_dim, action_dim).to(device)
    medical_ode = MedicalODE().to(device)
    neural_ode  = NeuralODE().to(device)

    data = patient["data"].clone()
    mask = patient["mask"]
    dt   = patient["dt"]

    dxdt_clip = compute_dxdt_clip_vals(data, mask, dt)

    # Scale rollout_steps to the patient's sequence length:
    # cap at T//3 so there are at least 3× as many anchor points as rollout depth.
    T = data.shape[0]
    rollout_steps = max(2, min(30, T // 3))

    print("\n[Auto-computed dx/dt clipping range (normalized space /h)]")
    for f, c in zip(state_features, dxdt_clip):
        print(f"  {f:<12}: +/-{c:.3f}")
    print(f"[rollout_steps={rollout_steps}  (T={T})]")

    pinn_optim = optim.Adam(model.parameters(), lr=lr_pinn, weight_decay=1e-4)
    ode_optim  = optim.Adam(
        list(medical_ode.parameters()) + list(neural_ode.parameters()),
        lr=lr_ode, weight_decay=1e-3,
    )
    pinn_sched = optim.lr_scheduler.CosineAnnealingLR(pinn_optim, T_max=epochs, eta_min=1e-6)
    ode_sched  = optim.lr_scheduler.CosineAnnealingLR(ode_optim,  T_max=epochs, eta_min=1e-6)

    # Initialise best_state with the untrained weights so it is never None,
    # even if NaN is detected on the very first epoch.
    def _snapshot():
        return {
            k: {p: v.clone() for p, v in m.state_dict().items()}
            for k, m in [("model", model),
                         ("medical_ode", medical_ode),
                         ("neural_ode", neural_ode)]
        }

    best_loss, best_state, no_improve = float("inf"), _snapshot(), 0
    p2, p3 = epochs // 2, epochs * 2 // 3

    for ep in range(epochs):
        if ep < p2:
            w = dict(w_data=20., w_roll=40., w_ode=10, w_smooth=0.1)
        elif ep < p3:
            w = dict(w_data=10., w_roll=20., w_ode=20, w_smooth=0.05)
        else:
            w = dict(w_data=5., w_roll=20., w_ode=20, w_smooth=0.02)

        loss = pinn_loss(model, scales, medical_ode, neural_ode,
                         data, mask, dt, dxdt_clip,
                         rollout_steps=rollout_steps, **w)

        pinn_optim.zero_grad()
        ode_optim.zero_grad()
        loss.backward()

        if torch.isnan(loss):
            print(f"ep{ep}: NaN detected, stopping.")
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            list(medical_ode.parameters()) + list(neural_ode.parameters()),
            max_norm=0.5,
        )

        pinn_optim.step();  ode_optim.step()
        pinn_sched.step();  ode_sched.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = _snapshot()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stop ep={ep}, best={best_loss:.5f}")
            break

        if ep % 500 == 0:
            sc  = torch.nn.functional.softplus(
                model.log_scale).detach().cpu().numpy()
            tag = "P1" if ep < p2 else "P2" if ep < p3 else "P3"
            print(f"ep{ep:5d}[{tag}] loss={loss.item():.4f}  "
                  f"scales={sc.round(3)}")

    model.load_state_dict(best_state["model"])
    medical_ode.load_state_dict(best_state["medical_ode"])
    neural_ode.load_state_dict(best_state["neural_ode"])

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(),       f"{save_dir}/pinn.pt")
    torch.save(medical_ode.state_dict(), f"{save_dir}/medical_ode.pt")
    torch.save(neural_ode.state_dict(),  f"{save_dir}/neural_ode.pt")
    # Scales are saved as cluster_scales.npy by main.py (_enforce_physical_bounds)
    # using unified keys (mean/std/ascl/state_min/state_max). No separate scales.npz.
    print(f"\n[PINN] Saved to {save_dir}/  best_loss={best_loss:.6f}")

    return model, medical_ode, neural_ode, torch.tensor(best_loss)
