"""Evaluate ind / clu / pop LAG-TRPO policies on each ind (test) patient PINN.

  ind: ind-patient policy on ind-patient PINN, starting from ind t=0
  clu: clu-patient policy on ind-patient PINN, starting from ind t=0
  pop: population policy on ind-patient PINN, starting from ind t=0

Primary metrics: MAP (target 65-80 mmHg) + Lac (target < 2 mmol/L).

Outputs:
  results/hypotension/<algo>/K<K>/n<n_eval>_noise<std>/
    per_cluster/cluster_<cid>/
      rollout_<algo>_ind_<dt_mode>.npz
      rollout_<algo>_clu_<dt_mode>.npz
      rollout_<algo>_pop_<dt_mode>.npz
    manifest.csv
    _meta.json

Run from the medrl-tacos root:

    python scripts/hypotension/eval_hypo.py --dt_mode fixdt
    python scripts/hypotension/eval_hypo.py --dt_mode vardt --n_eval 20
    python scripts/hypotension/eval_hypo.py --dt_mode fixdt vardt --n_eval 10
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from hypotension.data.config import (PINN_IND_ROOT, PINN_POP_ROOT, ONLINE_ROOT,
                                      EVAL_ROOT, PROCESSED_ROOT, state_dim, action_dim)
from hypotension.data.loader import load_patients
from hypotension.models.pinn import PINN
from hypotension.env.environment import HypoICUEnvironment, SMDP_STATE_DIM, OPTION_ACTION_DIM
from hypotension.train.pinn.io_utils import enforce_pinn_physical_bounds, norm_bounds

from rl.lagrangian_trpo import LagrangianTRPO

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


# ── env builder ───────────────────────────────────────────────────────────────

def _build_env(ind_pid: int, use_dt: bool,
               processed: str, pinn_root: Path) -> tuple:
    patient, _, sc = load_patients(processed, ind_pid, 1.0)
    scales = enforce_pinn_physical_bounds(sc)
    sd  = torch.load(str(pinn_root / f"patient_{ind_pid}" / "pinn.pt"), map_location="cpu")
    hid = sd["net.0.weight"].shape[0]
    pinn = PINN(hidden=hid); pinn.load_state_dict(sd); pinn.eval()

    mean      = np.array(scales["mean"],      np.float32)
    std       = np.array(scales["std"],       np.float32)
    ascl_mean = np.array(scales["ascl_mean"], np.float32)
    ascl_std  = np.array(scales["ascl_std"],  np.float32)
    lo, hi    = norm_bounds(scales)
    a_min     = (np.array([0.0,  0.0,   0.21], np.float32) - ascl_mean) / ascl_std
    a_max     = (np.array([12.5, 0.254, 1.00], np.float32) - ascl_mean) / ascl_std

    env = HypoICUEnvironment(
        pinn_model=pinn, mean_np=mean, std_np=std,
        action_min_pinn=a_min, action_max_pinn=a_max,
        state_lo_norm=lo, state_hi_norm=hi, action_scale_np=ascl_std,
        max_steps=48, dt_min=1.0, dt_max=6.0, total_time_h=48.0,
        use_dt=use_dt, fixed_dt=1.0,
    )
    init = patient["data"][0, 1:1+state_dim].cpu().numpy().astype(np.float32)
    return env, init, scales


# ── policy loader ─────────────────────────────────────────────────────────────

def _load_policy(pt_path: Path) -> LagrangianTRPO:
    raw     = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    actor_sd = raw.get("actor", raw)
    first_key = next(k for k in actor_sd if "weight" in k)
    hidden  = actor_sd[first_key].shape[0]
    agent   = LagrangianTRPO(state_dim=SMDP_STATE_DIM, action_dim=OPTION_ACTION_DIM,
                              hidden=hidden)
    agent.actor.load_state_dict(actor_sd, strict=False)
    agent.actor.eval()
    return agent


# ── episode runner ────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_episode(agent, env, init: np.ndarray,
                 noise_std: float = 0.0) -> dict:
    obs = env.reset(init_state_norm=init)
    if noise_std > 0:
        obs = obs + np.random.normal(0, noise_std, obs.shape).astype(np.float32)
    maps, lacs, rewards, dts = [], [], [], []
    done = False
    while not done:
        action, *_ = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        if noise_std > 0:
            obs = obs + np.random.normal(0, noise_std, obs.shape).astype(np.float32)
        maps.append(info["map"])
        lacs.append(info["lactate"])
        rewards.append(reward)
        dts.append(info["dt"])
    return dict(maps=np.array(maps), lacs=np.array(lacs),
                rewards=np.array(rewards), dts=np.array(dts))


# ── main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dt_mode", nargs="+", default=["fixdt"], choices=["fixdt", "vardt"])
    p.add_argument("--algo", default="lagrangian_trpo")
    p.add_argument("--K", type=int, default=48)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--processed", default=str(PROCESSED_ROOT))
    p.add_argument("--pinn_ind_root", default=str(PINN_IND_ROOT))
    p.add_argument("--pinn_pop_root", default=str(PINN_POP_ROOT))
    p.add_argument("--online_root", default=str(ONLINE_ROOT))
    p.add_argument("--eval_root", default=str(EVAL_ROOT))
    return p.parse_args()


def main():
    a = _parse()
    t0 = time.time()

    for dt_mode in a.dt_mode:
        use_dt = (dt_mode == "vardt")
        tag = f"n{a.n_eval}_noise{str(a.noise_std).replace('.','p')}"
        out_root = Path(a.eval_root) / a.algo / f"K{a.K}" / tag

        for cid, pids in GROUPS.items():
            ind_pid = pids["ind"]
            clu_pid = pids["clu"]
            out_dir = out_root / "per_cluster" / f"cluster_{cid}"
            out_dir.mkdir(parents=True, exist_ok=True)

            env, init, _ = _build_env(ind_pid, use_dt,
                                       a.processed, Path(a.pinn_ind_root))

            for scope in ("ind", "clu", "pop"):
                if scope == "ind":
                    pt = (Path(a.online_root) / "ind" / f"patient_{ind_pid}"
                          / a.algo / f"K{a.K}" / dt_mode / "policy.pt")
                elif scope == "clu":
                    pt = (Path(a.online_root) / "clu" / f"cluster_{cid}"
                          / a.algo / f"K{a.K}" / dt_mode / "policy.pt")
                else:
                    pt = (Path(a.online_root) / "pop"
                          / a.algo / f"K{a.K}" / dt_mode / "policy.pt")

                if not pt.exists():
                    print(f"  [skip] {pt} not found")
                    continue

                agent = _load_policy(pt)
                rollouts = [_run_episode(agent, env, init, a.noise_std)
                            for _ in range(a.n_eval)]

                np.savez(out_dir / f"rollout_{a.algo}_{scope}_{dt_mode}.npz",
                         maps=np.stack([r["maps"] for r in rollouts]),
                         lacs=np.stack([r["lacs"] for r in rollouts]),
                         rewards=np.stack([r["rewards"] for r in rollouts]),
                         dts=np.stack([r["dts"] for r in rollouts]))

                mean_map = np.mean([r["maps"].mean() for r in rollouts])
                mean_lac = np.mean([r["lacs"].mean() for r in rollouts])
                print(f"  [{scope}] cid={cid} {dt_mode}  "
                      f"MAP={mean_map:.1f}  Lac={mean_lac:.2f}")

        meta = dict(algo=a.algo, K=a.K, dt_mode=dt_mode, n_eval=a.n_eval,
                    noise_std=a.noise_std, elapsed_s=round(time.time()-t0, 1))
        (out_root / "_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  [done] {dt_mode} -> {out_root}")


if __name__ == "__main__":
    main()
