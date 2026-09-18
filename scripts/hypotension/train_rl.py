"""Train LAG-TRPO / TRPO / LAG-PPO on the hypotension environment.

Three scopes, all using the same 10 cluster groups:

  ind  -- patient-specific PINN for each ind (test) patient
          PINN: checkpoints-hypotension/pinn/ind/patient_{ind_pid}/pinn.pt
          save: checkpoints-hypotension/online/ind/patient_{ind_pid}/{algo}/K{K}/{dt_mode}/

  clu  -- patient-specific PINN for each clu (nearest) patient
          PINN: checkpoints-hypotension/pinn/clu/cluster_{cid}/pinn.pt
          save: checkpoints-hypotension/online/clu/cluster_{cid}/{algo}/K{K}/{dt_mode}/

  pop  -- shared population PINN (trained on all 10 clu patients)
          PINN: checkpoints-hypotension/pinn/pop/clu10/pinn.pt
          save: checkpoints-hypotension/online/pop/{algo}/K{K}/{dt_mode}/

Run from the medrl-tacos root:

    # single run
    python scripts/hypotension/train_rl.py --scope ind --cluster_id 1 --dt_mode fixdt
    python scripts/hypotension/train_rl.py --scope pop --dt_mode vardt

    # full sweep (42 runs: 20 ind + 20 clu + 2 pop)
    bash scripts/hypotension/sweep_rl.sh
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

from hypotension.data.config import (PINN_IND_ROOT, PINN_CLU_ROOT, PINN_POP_ROOT, ONLINE_ROOT,
                                      PROCESSED_ROOT, state_dim, action_dim)
from hypotension.data.loader import load_patients
from hypotension.models.pinn import PINN
from hypotension.env.environment import HypoICUEnvironment, SMDP_STATE_DIM, OPTION_ACTION_DIM
from hypotension.train.pinn.io_utils import enforce_pinn_physical_bounds, norm_bounds
from hypotension.train.train_onpolicy import train_onpolicy_hypo

from rl.lagrangian_trpo import LagrangianTRPO
from rl.lagrangian_ppo  import LagrangianPPO
from rl.trpo            import TRPO

# ── cluster / group mapping ────────────────────────────────────────────────────
GROUPS: dict[int, dict[str, int]] = {
    1:  {"ind": 36, "clu": 26},
    2:  {"ind": 39, "clu":  7},
    3:  {"ind": 53, "clu": 52},
    4:  {"ind":  3, "clu": 59},
    5:  {"ind": 45, "clu": 43},
    6:  {"ind": 46, "clu": 32},
    7:  {"ind":  9, "clu":  4},
    8:  {"ind": 33, "clu": 57},
    9:  {"ind": 14, "clu": 35},
    10: {"ind": 34, "clu": 48},
}
CLU_PIDS = [GROUPS[i]["clu"] for i in sorted(GROUPS)]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_scales_for(pid: int, processed: str) -> dict:
    _, _, sc = load_patients(processed, pid, 1.0)
    return enforce_pinn_physical_bounds(sc)


def _load_pinn(pinn_pt: Path) -> PINN:
    sd  = torch.load(str(pinn_pt), map_location="cpu")
    hid = sd["net.0.weight"].shape[0]
    m   = PINN(hidden=hid)
    m.load_state_dict(sd)
    return m.eval()


def _build_env(pinn: PINN, scales: dict, max_steps: int, use_dt: bool,
               dt_min: float = 1.0, dt_max: float = 6.0) -> HypoICUEnvironment:
    mean      = np.array(scales["mean"],      np.float32)
    std       = np.array(scales["std"],       np.float32)
    ascl_mean = np.array(scales["ascl_mean"], np.float32)
    ascl_std  = np.array(scales["ascl_std"],  np.float32)
    lo, hi    = norm_bounds(scales)
    a_min = (np.array([0.0,  0.0,   0.21], np.float32) - ascl_mean) / ascl_std
    a_max = (np.array([12.5, 0.254, 1.00], np.float32) - ascl_mean) / ascl_std
    return HypoICUEnvironment(
        pinn_model=pinn, mean_np=mean, std_np=std,
        action_min_pinn=a_min, action_max_pinn=a_max,
        state_lo_norm=lo, state_hi_norm=hi, action_scale_np=ascl_std,
        max_steps=max_steps, dt_min=dt_min, dt_max=dt_max,
        total_time_h=48.0, use_dt=use_dt, fixed_dt=dt_min,
    )


def _make_agent(algo: str, hidden: int, delta: float, lr_critic: float,
                cost_limit: float, lr_lagrange: float):
    kw = dict(state_dim=SMDP_STATE_DIM, action_dim=OPTION_ACTION_DIM, hidden=hidden)
    if algo == "lagrangian_trpo":
        return LagrangianTRPO(**kw, delta=delta, lr_critic=lr_critic,
                              cost_limit=cost_limit, lr_lagrange=lr_lagrange)
    if algo == "lagrangian_ppo":
        return LagrangianPPO(**kw, lr_critic=lr_critic,
                             cost_limit=cost_limit, lr_lagrange=lr_lagrange)
    if algo == "trpo":
        return TRPO(**kw, delta=delta, lr_critic=lr_critic)
    raise ValueError(f"Unknown algo: {algo}")


# ── main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scope", choices=["ind", "clu", "pop"], required=True)
    p.add_argument("--cluster_id", type=int, default=None,
                   help="Cluster ID 1-10 (required for ind/clu)")
    p.add_argument("--dt_mode", choices=["fixdt", "vardt"], default="fixdt")
    p.add_argument("--algo", default="lagrangian_trpo")
    p.add_argument("--K", type=int, default=48)
    p.add_argument("--total_steps", type=int, default=400_000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--delta", type=float, default=0.02)
    p.add_argument("--lr_critic", type=float, default=3e-3)
    p.add_argument("--cost_limit", type=float, default=0.8)
    p.add_argument("--lr_lagrange", type=float, default=0.05)
    p.add_argument("--dt_min", type=float, default=1.0)
    p.add_argument("--dt_max", type=float, default=6.0)
    p.add_argument("--rollout_len", type=int, default=2048)
    p.add_argument("--processed", default=str(PROCESSED_ROOT))
    p.add_argument("--pinn_ind_root", default=str(PINN_IND_ROOT))
    p.add_argument("--pinn_clu_root", default=str(PINN_CLU_ROOT))
    p.add_argument("--pinn_pop_root", default=str(PINN_POP_ROOT))
    p.add_argument("--online_root", default=str(ONLINE_ROOT))
    return p.parse_args()


def main():
    a = _parse()
    use_dt = (a.dt_mode == "vardt")

    if a.scope in ("ind", "clu"):
        if a.cluster_id is None:
            raise SystemExit("--cluster_id is required for scope=ind/clu")
        pid = GROUPS[a.cluster_id][a.scope]
        if a.scope == "clu":
            pinn_pt = Path(a.pinn_clu_root) / f"cluster_{a.cluster_id}" / "pinn.pt"
        else:
            pinn_pt = Path(a.pinn_ind_root) / f"patient_{pid}" / "pinn.pt"
        if a.scope == "clu":
            save_dir = (Path(a.online_root) / "clu" / f"cluster_{a.cluster_id}"
                        / a.algo / f"K{a.K}" / a.dt_mode)
        else:
            save_dir = (Path(a.online_root) / "ind" / f"patient_{pid}"
                        / a.algo / f"K{a.K}" / a.dt_mode)
    else:  # pop
        pinn_pt = Path(a.pinn_pop_root) / "clu10" / "pinn.pt"
        save_dir = Path(a.online_root) / "pop" / a.algo / f"K{a.K}" / a.dt_mode

    print(f"[{a.scope}] PINN: {pinn_pt}")
    print(f"[{a.scope}] save: {save_dir}")

    if a.scope == "pop":
        pid_for_scales = CLU_PIDS[0]
    elif a.scope == "clu":
        pid_for_scales = GROUPS[a.cluster_id]["clu"]
    else:
        pid_for_scales = GROUPS[a.cluster_id]["ind"]

    scales = _load_scales_for(pid_for_scales, a.processed)
    pinn   = _load_pinn(pinn_pt)
    env    = _build_env(pinn, scales, a.K, use_dt, a.dt_min, a.dt_max)
    agent  = _make_agent(a.algo, a.hidden, a.delta, a.lr_critic,
                         a.cost_limit, a.lr_lagrange)

    if a.scope == "pop":
        inits = []
        for pid in CLU_PIDS:
            pt, _, _ = load_patients(a.processed, pid, 1.0)
            inits.append(pt["data"][0, 1:1 + state_dim].cpu().numpy())
        init_state = np.stack(inits)
    else:
        pt, _, _ = load_patients(a.processed, pid, 1.0)
        init_state = pt["data"][0, 1:1 + state_dim].cpu().numpy()

    agent_name = f"{a.algo.upper()}_{a.dt_mode}"
    train_onpolicy_hypo(
        agent=agent, env=env, init_state_norm=init_state,
        total_steps=a.total_steps, rollout_len=a.rollout_len,
        save_dir=str(save_dir), agent_name=agent_name,
    )


if __name__ == "__main__":
    main()
