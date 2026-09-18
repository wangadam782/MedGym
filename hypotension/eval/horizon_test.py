"""1-step / H-step horizon evaluation for the hypotension PINN.

One-step test (teacher forcing):
    At each time t, feed REAL x(t) and u(t) to the PINN, get x_hat(t+1),
    compare with real x(t+1). Each step starts from a real state; errors do not accumulate.

H-step test (open-loop rollout, H=5):
    From REAL x(t), roll the PINN forward H steps using real actions u(t)..u(t+H-1)
    (without resetting to real states). Compare x_hat(t+H) with real x(t+H).
    This is the true error when using the PINN as a simulator.

Three mandatory baselines (a simulator that cannot beat all three has not learned dynamics):
  persistence  x_hat(t+H) = x(t)              "nothing changes"
  patient_mean x_hat(t+H) = patient mean       "ignore time"
  cohort_mean  x_hat(t+H) = cohort mean (z=0) "know nothing"

Metrics are computed in normalised space; 1 unit = 1 cohort standard deviation = nMAE.
"""

from __future__ import annotations

import numpy as np
import torch

from hypotension.data.config import STATE_NAMES, device, state_dim
from hypotension.data.loader import split_data
from hypotension.train.pinn.io_utils import norm_bounds


# =============================================================================
def _starts_for(patient, H, split):
    """Valid starting points. split='test' uses only out-of-sample segment."""
    T = patient.get("T", patient["data"].shape[0])
    n_tr = patient.get("n_train", T)
    all_s = np.arange(0, T - H)
    if split == "train":
        return all_s[all_s + H < n_tr]
    if split == "test":
        return all_s[all_s >= n_tr]
    return all_s


@torch.no_grad()
def horizon_predictions(pinn, patient, scales, H: int, n_substeps: int = 8,
                        split: str = "test"):
    """Return H-step predictions and ground truth.

    Returns dict with keys:
        starts: (N,)          valid starting points
        pred  : (N, d_x)      PINN H-step prediction from x(start)
        true  : (N, d_x)      real x(start+H)
        mask  : (N, d_x)      observation mask at x(start+H)
        traj  : (N, H+1, d_x) full predicted trajectory
    """
    _, x, u = split_data(patient)
    m = patient["mask"]
    lo_np, hi_np = norm_bounds(scales)
    lo = torch.as_tensor(lo_np, device=device)
    hi = torch.as_tensor(hi_np, device=device)
    dt = float(scales.get("dt", 1.0))

    starts = _starts_for(patient, H, split)
    if len(starts) == 0:
        raise ValueError(f"split={split} H={H}: no valid starting points "
                         f"(trajectory length {x.shape[0]}, n_train={patient.get('n_train')})"
                         f"\n  -> reduce --train_frac or --horizons")
    xs = x[starts]
    traj = [xs]
    for j in range(H):
        xs = pinn.step(xs, u[starts + j], dt, lo, hi, n_substeps)
        traj.append(xs)
    return dict(starts=starts, pred=xs, true=x[starts + H], mask=m[starts + H],
                traj=torch.stack(traj, 1), x_all=x, m_all=m)


@torch.no_grad()
def baselines(patient, H: int, starts, split='test'):
    """Three trivial baselines for H-step prediction (normalised space)."""
    _, x, _ = split_data(patient)
    n_tr = patient.get("n_train", x.shape[0])
    pm = x[:n_tr].mean(0, keepdim=True)
    return dict(
        persistence=x[starts],
        patient_mean=pm.expand(len(starts), -1),
        cohort_mean=torch.zeros(len(starts), x.shape[1], device=x.device),
    )


def _agg(err, mask):
    """Mask-weighted MAE / RMSE, per variable."""
    w = (mask >= 1.0).float()
    n = w.sum(0).clamp(min=1)
    mae = (err.abs() * w).sum(0) / n
    rmse = (((err ** 2) * w).sum(0) / n).sqrt()
    return mae.cpu().numpy(), rmse.cpu().numpy(), (w.sum(0) > 0).cpu().numpy()


@torch.no_grad()
def evaluate_horizon(pinn, patient, scales, H, n_substeps=8, split="test"):
    """Full evaluation for one patient, one H. split: 'test' / 'train' / 'all'."""
    r = horizon_predictions(pinn, patient, scales, H, n_substeps, split)
    bl = baselines(patient, H, r["starts"], split)
    mask = r["mask"]

    res = {}
    mae, rmse, obs = _agg(r["pred"] - r["true"], mask)
    res["pinn"] = dict(mae=mae, rmse=rmse)
    for k, v in bl.items():
        m2, r2, _ = _agg(v - r["true"], mask)
        res[k] = dict(mae=m2, rmse=r2)
    res["observed"] = obs
    res["n_starts"] = len(r["starts"])
    res["split"] = split

    s0 = int(r["starts"][0])
    res["from_t0"] = dict(
        t0=s0,
        pred=r["traj"][0].cpu().numpy(),
        true=r["x_all"][s0: s0 + H + 1].cpu().numpy(),
        mask=r["m_all"][s0: s0 + H + 1].cpu().numpy(),
    )
    res["traj"] = r["traj"].cpu().numpy()
    res["true_seq"] = r["true"].cpu().numpy()
    res["pred_seq"] = r["pred"].cpu().numpy()
    res["starts"] = r["starts"]
    return res


# =============================================================================
def print_table(res: dict, H: int, pid, scales=None):
    obs = res["observed"]
    print(f"\n{'='*84}")
    print(f"  {H}-step test   patient {pid}   [{res.get('split','?')}]   "
          f"n_starts={res['n_starts']}   (metric = nMAE, 1.0 = 1 cohort std; real obs only)")
    print(f"{'='*84}")
    print(f"{'var':<7}{'PINN':>10}{'persistence':>14}{'patient_mean':>15}"
          f"{'cohort_mean':>14}   result")
    print("-" * 84)
    for j, n in enumerate(STATE_NAMES):
        if not obs[j]:
            print(f"{n:<7}{'—':>10}{'—':>14}{'—':>15}{'—':>14}   (no real obs)")
            continue
        p = res["pinn"]["mae"][j]
        pe = res["persistence"]["mae"][j]
        pm = res["patient_mean"]["mae"][j]
        cm = res["cohort_mean"]["mae"][j]
        best = min(pe, pm, cm)
        tag = "beats all baselines" if p < best else (
            f"lost to {['persistence','patient_mean','cohort_mean'][int(np.argmin([pe,pm,cm]))]}")
        print(f"{n:<7}{p:>10.4f}{pe:>14.4f}{pm:>15.4f}{cm:>14.4f}   {tag}")
    print("-" * 84)
    sel = obs
    print(f"{'mean':<7}{res['pinn']['mae'][sel].mean():>10.4f}"
          f"{res['persistence']['mae'][sel].mean():>14.4f}"
          f"{res['patient_mean']['mae'][sel].mean():>15.4f}"
          f"{res['cohort_mean']['mae'][sel].mean():>14.4f}")


def print_from_t0(res: dict, H: int, scales, keys=("MAP", "Lac", "UO", "PP")):
    """Step-by-step comparison from the first real state (physical units)."""
    mean, std = scales["mean"], scales["std"]
    pred = res["from_t0"]["pred"] * std + mean
    true = res["from_t0"]["true"] * std + mean
    t0 = res["from_t0"].get("t0", 0)
    print(f"\n  From real state at t={t0}, step-by-step comparison (physical units)")
    hdr = "  step" + "".join(f"{k+'(true)':>11}{k+'(pred)':>11}" for k in keys)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in range(H + 1):
        row = f"  {s:>4}"
        for k in keys:
            j = STATE_NAMES.index(k)
            row += f"{true[s, j]:>11.2f}{pred[s, j]:>11.2f}"
        print(row)
