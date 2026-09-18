"""_eval_base.py — Per-patient (Ind) vs population (Pop) rollouts for one algorithm.

For ``--algo`` in {sac, ppo, trpo, lagrangian_ppo, lagrangian_trpo}, evaluates
four policies on each patient's PINN-backed ``ICUEnvironment``:

  Ind fixdt / Ind vardt — weights under ``--ind_root`` (default
    ``results/online/individual``): ``patient_<id>/<algo>/K<K>/<tag>/policy.pt``.
  Pop fixdt / Pop vardt — weights under ``--pop_root`` (default
    ``results/online/population``): ``<algo>/K<K>/<tag>/policy.pt``,
    wrapped in ``AdaptedEnv`` using ``--glb_scales`` (default
    ``results/pinn/population/scales.npy``). Per-patient env scales come from
    each patient's PINN directory ``scales.npy`` (next to ``pinn.pt``).
    Each patient directory must also contain ``init_state_norm.npy`` (no MIMIC
    trajectory CSV is read).

Loads actor weights flexibly (nested ``actor`` dict vs flat state_dict; full
checkpoint fallback). Outputs default to ``--save_root`` (default
``results/online/transfer_eval``) unless ``--save_dir`` is set.

Usage:
  python scripts/_eval_base.py --algo sac --K 20 --n_eval 5

  python scripts/_eval_base.py --algo trpo \\
      --pinn_dir results/pinn/individual \\
      --ind_root results/online/individual \\
      --pop_root results/online/population \\
      --glb_scales results/pinn/population/scales.npy

  python scripts/_eval_base.py --algo sac --plot_only --save_dir <prior_run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from data.config import state_dim, action_dim
from data.config import PINN_IND_ROOT, POLICY_IND_ROOT, POLICY_POP_ROOT, PINN_POP_ROOT, POLICY_TRANSFER_EVAL_ROOT
from utils.pinn_scales import (
    load_individual_init_state_norm,
    load_pinn_folder_scales,
)


# ─── Constants ────────────────────────────────────────────────────────────────
POST_HOUR        = 20.0
TOTAL_TIME_H     = 96.0
T_GRID           = np.arange(0, TOTAL_TIME_H + 1, 1.0)
LAC_DANGER_THR   = 4.0
LAC_NORMAL_THR   = 2.0

PHYSICAL_MIN = np.array([0.,   0.,   0.,  3.,    0.,  0. ], dtype=np.float32)
PHYSICAL_MAX = np.array([100., 600., 30., 15., 5000., 20.], dtype=np.float32)

COLOR_IND = "#D62728"  # tab:red
COLOR_POP = "#1F77B4"  # tab:blue


# ─── Algorithm registry ──────────────────────────────────────────────────────
# Per-algorithm metadata.  ``cls_name`` is resolved lazily inside the worker
# (so ``import torch`` / ``rl`` happens only when actually running).

ALGO_REG: dict[str, dict] = {
    "sac": {
        "cls_name":         "SAC",
        "file_prefix":      "sac",
        "label":            "SAC",
        "default_ind_root": "results/online/individual",
        "default_hidden":   256,
    },
    "ppo": {
        "cls_name":         "PPO",
        "file_prefix":      "ppo",
        "label":            "PPO",
        "default_ind_root": "results/online/individual",
        "default_hidden":   256,
    },
    "trpo": {
        "cls_name":         "TRPO",
        "file_prefix":      "trpo",
        "label":            "TRPO",
        "default_ind_root": "results/online/individual",
        "default_hidden":   256,
    },
    "lagrangian_ppo": {
        "cls_name":         "LagrangianPPO",
        "file_prefix":      "lagrangian_ppo",
        "label":            "PPOLag",
        "default_ind_root": "results/online/individual",
        "default_hidden":   256,
    },
    "lagrangian_trpo": {
        "cls_name":         "LagrangianTRPO",
        "file_prefix":      "lagrangian_trpo",
        "label":            "TRPOLag",
        "default_ind_root": "results/online/individual",
        "default_hidden":   256,
    },
}


def _policies_for(algo: str) -> list[str]:
    return [f"{algo}_ind_fixdt", f"{algo}_ind_vardt",
            f"{algo}_pop_fixdt", f"{algo}_pop_vardt"]


def _labels_for(algo: str) -> dict[str, str]:
    pretty = ALGO_REG[algo]["label"]
    return {
        f"{algo}_ind_fixdt": f"{pretty}-F-Ind",
        f"{algo}_ind_vardt": f"{pretty}-A-Ind",
        f"{algo}_pop_fixdt": f"{pretty}-F-Pop",
        f"{algo}_pop_vardt": f"{pretty}-A-Pop",
    }


def _colors_for(algo: str) -> dict[str, str]:
    return {
        f"{algo}_ind_fixdt": COLOR_IND,
        f"{algo}_ind_vardt": COLOR_IND,
        f"{algo}_pop_fixdt": COLOR_POP,
        f"{algo}_pop_vardt": COLOR_POP,
    }


def _lighten_toward_white(hex_or_rgb, frac: float = 0.58) -> tuple:
    r, g, b = mcolors.to_rgb(hex_or_rgb)
    return tuple(c + (1.0 - c) * frac for c in (r, g, b))


# ─── Path resolvers ──────────────────────────────────────────────────────────

def _resolve_pinn_path(pinn_root: Path, pid: int) -> Path | None:
    """Pick ``pinn.pt`` from a directory that also has ``scales.npy`` when possible.

    Prefer ``<pinn_root>/<pid>/`` over ``patient_<pid>/`` so numeric layouts
    (where artifacts usually live) win when both exist.
    """
    dirs = [
        pinn_root / str(pid),
        pinn_root / f"patient_{pid}",
        pinn_root / str(pid) / f"patient_{pid}",
    ]
    first_pt: Path | None = None
    for d in dirs:
        pt = d / "pinn.pt"
        if not pt.exists():
            continue
        if first_pt is None:
            first_pt = pt
        if (d / "scales.npy").exists() or (d / "cluster_scales.npy").exists():
            return pt
    return first_pt


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _resolve_ind_path(ind_root: Path, pid: int, tag: str, K: int,
                      file_prefix: str) -> Path | None:
    """Per-patient weight path; tries several historical layouts.

    Layouts tried (first match wins):
        <ind_root>/patient_<pid>/<tag>/<file_prefix>_<tag>.pt
        <ind_root>/patient_<pid>/<file_prefix>_<tag>_K<K>/<file_prefix>_<tag>.pt
    """
    base = ind_root / f"patient_{pid}" / file_prefix / f"K{K}" / tag
    candidates = [
        base / "policy.pt",
        base / "actor.pt",
        ind_root / f"patient_{pid}" / tag / f"{file_prefix}_{tag}.pt",
        ind_root / f"patient_{pid}" / f"{file_prefix}_{tag}_K{K}" / f"{file_prefix}_{tag}.pt",
    ]
    return _first_existing(
        *candidates,
    )


def _resolve_pop_path(pop_root: Path, tag: str, K: int,
                      file_prefix: str) -> Path | None:
    """Population (global) weight path."""
    base = pop_root / file_prefix / f"K{K}" / tag

    candidates = [
        base / "policy.pt",
        base / "actor.pt",
        pop_root / f"{file_prefix}_{tag}_K{K}" / f"{file_prefix}_{tag}.pt",
    ]
    return _first_existing(
        *candidates,
    )

def _save_meta(args, pids, save_dir: Path):
    meta = {
        "script": "scripts/_eval_base.py",
        "task": "transfer_eval_ind_cluster_pop",
        "algo": args.algo,
        "K": int(args.K),
        "n_eval": int(args.n_eval),
        "init_noise_std": float(args.init_noise_std),
        "init_source": "init_state_norm.npy_only",
        "pinn_dir": str(Path(args.pinn_dir).resolve()),
        "ind_root": str(Path(args.ind_root).resolve()),
        "pop_root": str(Path(args.pop_root).resolve()),
        "glb_scales": str(Path(args.glb_scales).resolve()),
        "save_dir": str(Path(save_dir).resolve()),
        "patient_ids": [int(x) for x in pids],
    }
    with open(Path(save_dir) / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ─── State-dict helpers ──────────────────────────────────────────────────────

def _extract_actor_sd(raw):
    if isinstance(raw, dict) and "actor" in raw and isinstance(raw["actor"], dict):
        return raw["actor"]
    return raw


def _infer_hidden(actor_sd) -> int | None:
    w = actor_sd.get("net.0.weight")
    return int(w.shape[0]) if w is not None and hasattr(w, "shape") else None


def _infer_action_dim(actor_sd) -> int | None:
    w = actor_sd.get("mu_layer.weight")
    return int(w.shape[0]) if w is not None and hasattr(w, "shape") else None


# ─── AdaptedEnv (state + action scale conversion for global → patient) ──────
#
# Mirrors scripts/log.py and scripts/eval_sac_vs_lagrangian_trpo.py.  Wraps
# a per-patient ICUEnvironment so that a globally-trained agent observes
# the global-norm state and emits a global-norm action; the action is then
# converted back to the patient-norm scale before being applied to the
# patient PINN.
class AdaptedEnv:
    def __init__(self, patient_env,
                 pat_mean: np.ndarray, pat_std: np.ndarray, pat_ascl: np.ndarray,
                 gl_mean:  np.ndarray, gl_std:  np.ndarray, gl_ascl:  np.ndarray):
        self.env       = patient_env
        self.pat_mean  = np.asarray(pat_mean, dtype=np.float32)
        self.pat_std   = np.asarray(pat_std,  dtype=np.float32)
        self.pat_ascl  = np.asarray(pat_ascl, dtype=np.float32)
        self.gl_mean   = np.asarray(gl_mean,  dtype=np.float32)
        self.gl_std    = np.asarray(gl_std,   dtype=np.float32)
        self.gl_ascl   = np.asarray(gl_ascl,  dtype=np.float32)
        self.use_dt    = patient_env.use_dt
        self.smdp_state_dim    = patient_env.smdp_state_dim
        self.option_action_dim = patient_env.option_action_dim
        self.dt_min            = patient_env.dt_min
        self.dt_max            = patient_env.dt_max
        self.total_time_h      = patient_env.total_time_h

    @property
    def current_t(self):
        return self.env.current_t

    def _obs_p2g(self, obs_pat: np.ndarray) -> np.ndarray:
        x_phys = obs_pat[:state_dim] * self.pat_std + self.pat_mean
        x_gl   = (x_phys - self.gl_mean) / self.gl_std
        return np.concatenate([x_gl, obs_pat[state_dim:]]).astype(np.float32)

    def _action_g2p(self, action_gl: np.ndarray) -> np.ndarray:
        u_tanh_gl = action_gl[:action_dim]
        u_norm_gl = np.clip((u_tanh_gl + 1.0) / 2.0, 0.0, 1.0)
        a_real    = np.expm1(u_norm_gl * self.gl_ascl)
        a_real    = np.clip(a_real, 0.0, None)
        u_norm_pt = np.log1p(a_real) / self.pat_ascl
        u_tanh_pt = np.clip(u_norm_pt * 2.0 - 1.0, -1.0, 1.0)
        if self.use_dt and len(action_gl) > action_dim:
            return np.concatenate([u_tanh_pt, action_gl[action_dim:]]).astype(np.float32)
        return u_tanh_pt.astype(np.float32)

    def reset(self, patient=None, init_state_norm=None):
        obs_pat = self.env.reset(patient=patient, init_state_norm=init_state_norm)
        return self._obs_p2g(obs_pat)

    def step(self, action_gl):
        action_pat = self._action_g2p(action_gl)
        obs_pat, reward, done, info = self.env.step(action_pat)
        return self._obs_p2g(obs_pat), reward, done, info


# ─── Rollout helpers ─────────────────────────────────────────────────────────

def _interp_zoh(t_arr, val_arr, t_grid):
    out = np.empty(len(t_grid), dtype=np.float32)
    for gi, tg in enumerate(t_grid):
        idx = max(0, int(np.searchsorted(t_arr, tg, side="right")) - 1)
        out[gi] = val_arr[min(idx, len(val_arr) - 1)]
    return out


def _run_episode_traj(agent, env,
                      init_norm_pat=None, init_noise_std=0.0, rng=None):
    if init_norm_pat is None:
        init_norm_pat = np.zeros(state_dim, dtype=np.float32)
    if init_noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng(42)
        noise   = rng.normal(0.0, init_noise_std, init_norm_pat.shape).astype(np.float32)
        init_pt = (init_norm_pat + noise).astype(np.float32)
        state   = env.reset(init_state_norm=init_pt)
    else:
        state = env.reset(init_state_norm=np.asarray(init_norm_pat, dtype=np.float32))

    done = False
    t_list, sofa_list, lac_list, dt_list = [], [], [], []

    while not done:
        t0     = float(env.current_t)
        action = agent.select_action(state, deterministic=True)
        state, _r, done, info = env.step(action)
        t1     = float(env.current_t)
        t_list.append(t0)
        sofa_list.append(float(info["sofa"]))
        lac_list.append(float(info["lactate"]))
        dt_list.append(t1 - t0)

    t_arr    = np.array(t_list,    dtype=np.float32)
    sofa_arr = np.array(sofa_list, dtype=np.float32)
    lac_arr  = np.array(lac_list,  dtype=np.float32)
    dt_arr   = np.array(dt_list,   dtype=np.float32)

    sofa_interp = _interp_zoh(t_arr, sofa_arr, T_GRID)
    lac_interp  = _interp_zoh(t_arr, lac_arr,  T_GRID)

    post_mask     = T_GRID >= POST_HOUR
    mean_sofa     = float(sofa_interp[post_mask].mean())
    mean_sofa_all = float(sofa_interp.mean())
    final_sofa    = float(sofa_interp[-1])
    mean_lac      = float(lac_interp.mean())
    safety_rate   = float(np.mean(lac_interp < LAC_DANGER_THR))

    lac_t0 = float(lac_interp[0])
    t6_idx = int(np.searchsorted(T_GRID, 6.0))
    lac_t6 = float(lac_interp[min(t6_idx, len(lac_interp) - 1)])
    clearance_6h = float((lac_t0 - lac_t6) / lac_t0) if lac_t0 > 0 else 0.0
    time_above_4 = float(np.sum(lac_interp > LAC_DANGER_THR))

    metrics = {
        "mean_sofa":     mean_sofa,
        "mean_sofa_all": mean_sofa_all,
        "final_sofa":    final_sofa,
        "mean_lac":      mean_lac,
        "safety_rate":   safety_rate,
        "clearance_6h":  clearance_6h,
        "time_above_4":  time_above_4,
        "mean_dt":       float(dt_arr.mean()),
    }
    return sofa_interp, lac_interp, metrics


def _collect_policy(agent, env, n_eval, init_norm_pat, init_noise_std, rng):
    sofa_trajs, lac_trajs = [], []
    keys = ["mean_sofa", "mean_sofa_all", "final_sofa",
            "mean_lac", "safety_rate", "clearance_6h", "time_above_4", "mean_dt"]
    all_metrics = {k: [] for k in keys}

    for _ in range(n_eval):
        sofa_t, lac_t, m = _run_episode_traj(
            agent, env,
            init_norm_pat=init_norm_pat, init_noise_std=init_noise_std, rng=rng,
        )
        sofa_trajs.append(sofa_t); lac_trajs.append(lac_t)
        for k in keys:
            all_metrics[k].append(m[k])

    sofa_mat = np.stack(sofa_trajs, axis=0)   # (n_eval, T_GRID)
    lac_mat  = np.stack(lac_trajs,  axis=0)   # (n_eval, T_GRID)
    return (sofa_mat.mean(0), sofa_mat.std(0),
            lac_mat.mean(0),  lac_mat.std(0),
            {k: float(np.mean(v)) for k, v in all_metrics.items()},
            sofa_mat, lac_mat,
            {k: np.asarray(v, dtype=np.float32) for k, v in all_metrics.items()})


# ─── Per-patient worker ───────────────────────────────────────────────────────

def _worker(pid, algo, pinn_dir, ind_root, pop_root, glb_scales_path,
            n_eval, K, init_noise_std=0.0):
    import torch
    import rl as _rl
    from models import PINN
    from rl import ICUEnvironment, SMDP_STATE_DIM, OPTION_ACTION_DIM
    from data.config import device as global_device

    device       = global_device
    cfg          = ALGO_REG[algo]
    AgentCls     = getattr(_rl, cfg["cls_name"])
    file_prefix  = cfg["file_prefix"]
    default_h    = cfg["default_hidden"]

    # ── Global (shared) scales ────────────────────────────────────────────
    gl_sc   = np.load(glb_scales_path, allow_pickle=True).item()
    gl_mean = gl_sc["mean"].astype(np.float32)
    gl_std  = gl_sc["std"].astype(np.float32)
    gl_ascl = gl_sc["ascl"].astype(np.float32)

    # ── Per-patient PINN (need folder for scales.npy) ─────────────────────
    pinn_path = _resolve_pinn_path(Path(pinn_dir), pid)
    if pinn_path is None:
        print(f"[patient {pid}] PINN not found under {pinn_dir}/{pid}/, skip.",
              flush=True)
        return None

    try:
        pat_mean, pat_std, pat_ascl, state_min_np, state_max_np, _ = load_pinn_folder_scales(
            pinn_path.parent, PHYSICAL_MIN, PHYSICAL_MAX
        )
    except FileNotFoundError as e:
        print(f"[patient {pid}] {e}, skipping.", flush=True)
        return None

    x_file = load_individual_init_state_norm(
        pinn_path.parent,
        state_dim=state_dim,
        state_min_np=state_min_np,
        state_max_np=state_max_np,
    )
    if x_file is None:
        print(
            f"[patient {pid}] missing init_state_norm.npy under {pinn_path.parent}; "
            "skipping (MIMIC trajectory CSV is no longer used here).",
            flush=True,
        )
        return None
    init_norm_pat = np.asarray(x_file, dtype=np.float32)
    pinn = PINN(state_dim, action_dim).to(device)
    pinn.load_state_dict(torch.load(str(pinn_path), map_location=device))
    pinn.eval()

    fixed_dt = TOTAL_TIME_H / K

    def make_patient_env(use_dt):
        return ICUEnvironment(
            pinn_model      = pinn,
            mean_np         = pat_mean,
            std_np          = pat_std,
            action_min_norm = np.zeros(action_dim, dtype=np.float32),
            action_max_norm = np.ones(action_dim,  dtype=np.float32),
            state_min       = state_min_np,
            state_max       = state_max_np,
            action_scale_np = pat_ascl,
            max_steps       = K,
            dt_min          = 0.5,
            dt_max          = 36.0,
            total_time_h    = TOTAL_TIME_H,
            use_dt          = use_dt,
            fixed_dt        = fixed_dt,
            use_lac_penalty = True,
        )

    rng     = np.random.default_rng(42)
    results = {}

    def _store(key, agent, env):
        sofa_mean, sofa_std, lac_mean, lac_std, m, \
            sofa_mat, lac_mat, per_ep_metrics = _collect_policy(
            agent, env, n_eval,
            init_norm_pat=init_norm_pat, init_noise_std=init_noise_std, rng=rng,
        )
        results[key] = {
            "sofa_mean": sofa_mean, "sofa_std": sofa_std,
            "lac_mean":  lac_mean,  "lac_std":  lac_std,
            "sofa_mat":  sofa_mat,   # (n_eval, T_GRID) — raw episode trajectories
            "lac_mat":   lac_mat,
            "per_ep":    per_ep_metrics,
            **m,
        }
        print(
            f"[patient {pid}] {key:<24} "
            f"sofa20={m['mean_sofa']:.2f}  "
            f"sofa_all={m['mean_sofa_all']:.2f}  "
            f"sofa96={m['final_sofa']:.2f}  "
            f"lac={m['mean_lac']:.2f}  "
            f"safe={m['safety_rate']:.2%}  "
            f"dt={m['mean_dt']:.2f}h",
            flush=True,
        )

    def _build_agent(pt_path: Path):
        """Try actor-only load with auto-inferred sizes; fall back to full load."""
        raw = torch.load(str(pt_path), map_location=device, weights_only=False)
        actor_sd = _extract_actor_sd(raw)
        hidden_use = _infer_hidden(actor_sd) or default_h
        adim_use   = _infer_action_dim(actor_sd) or OPTION_ACTION_DIM
        agent = AgentCls(state_dim=SMDP_STATE_DIM, action_dim=adim_use, hidden=hidden_use)
        try:
            agent.actor.load_state_dict(actor_sd)
            return agent
        except Exception as e:
            print(
                f"[patient {pid}] actor-only load failed ({e}); "
                f"falling back to agent.load() for {pt_path.name}",
                flush=True,
            )
            agent.load(str(pt_path))
            return agent

    # ── Ind: per-patient policy on per-patient env ───────────────────────
    for tag, use_dt in [("fixdt", False), ("vardt", True)]:
        key = f"{algo}_ind_{tag}"
        pt_path = _resolve_ind_path(Path(ind_root), pid, tag, K, file_prefix)
        if pt_path is None:
            print(
                f"[patient {pid}] {key}: model not found under "
                f"{ind_root}/patient_{pid}/ "
                f"(tried {tag}/ and {file_prefix}_{tag}_K{K}/)",
                flush=True,
            )
            results[key] = None
            continue
        agent = _build_agent(pt_path)
        _store(key, agent, make_patient_env(use_dt))

    # ── Pop: population policy on per-patient env via AdaptedEnv ─────────
    for tag, use_dt in [("fixdt", False), ("vardt", True)]:
        key = f"{algo}_pop_{tag}"
        pt_path = _resolve_pop_path(Path(pop_root), tag, K, file_prefix)
        if pt_path is None:
            print(
                f"[patient {pid}] {key}: model not found "
                f"({pop_root}/{file_prefix}_{tag}_K{K}/)",
                flush=True,
            )
            results[key] = None
            continue
        agent = _build_agent(pt_path)
        pat_env = make_patient_env(use_dt)
        env     = AdaptedEnv(pat_env, pat_mean, pat_std, pat_ascl,
                             gl_mean, gl_std, gl_ascl)
        _store(key, agent, env)

    return results


# ─── Trajectory I/O ──────────────────────────────────────────────────────────

def _save_trajectories(pid: int, results: dict, save_dir: str, algo: str) -> None:
    """Save per-episode trajectories for each of the 4 policies into
    ``<save_dir>/<pid>/rollout_<policy>.npz``.

    Each .npz contains:
      t_grid      : (T,)
      sofa_mat    : (n_eval, T)   raw per-episode SOFA trajectory
      lac_mat     : (n_eval, T)   raw per-episode lactate trajectory
      sofa_mean / sofa_std / lac_mean / lac_std : aggregates over episodes
      per_ep_<metric>            : (n_eval,)  per-episode scalar metrics
    """
    pid_dir = os.path.join(save_dir, str(pid))
    os.makedirs(pid_dir, exist_ok=True)
    for key in _policies_for(algo):
        r = results.get(key)
        if r is None or "sofa_mat" not in r:
            continue
        per_ep = r.get("per_ep", {})
        np.savez(
            os.path.join(pid_dir, f"rollout_{key}.npz"),
            t_grid    = T_GRID.astype(np.float32),
            sofa_mat  = r["sofa_mat"].astype(np.float32),
            lac_mat   = r["lac_mat"].astype(np.float32),
            sofa_mean = r["sofa_mean"].astype(np.float32),
            sofa_std  = r["sofa_std"].astype(np.float32),
            lac_mean  = r["lac_mean"].astype(np.float32),
            lac_std   = r["lac_std"].astype(np.float32),
            **{f"per_ep_{k}": v for k, v in per_ep.items()},
        )


# ─── Per-patient plots ───────────────────────────────────────────────────────

def _plot_per_patient(pid, results, save_dir, algo: str):
    os.makedirs(save_dir, exist_ok=True)
    LABELS = _labels_for(algo)
    COLORS = _colors_for(algo)

    for tag, title in [("fixdt", "fixed interval time"),
                       ("vardt", "adaptive interval time")]:
        keys = [f"{algo}_ind_{tag}", f"{algo}_pop_{tag}"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        for key in keys:
            if results.get(key) is None:
                continue
            mt = results[key]["sofa_mean"]; st = results[key]["sofa_std"]
            c  = COLORS[key]
            cf = _lighten_toward_white(c)
            ax.plot(T_GRID, mt, color=c, lw=2.0, label=LABELS[key])
            ax.fill_between(T_GRID, mt - st, mt + st, color=cf, alpha=0.35)
        ax.axvline(POST_HOUR, color="gray", ls="--", lw=1.0)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("SOFA score")
        ax.set_title(f"SOFA — {title}"); ax.legend()

        ax = axes[1]
        for key in keys:
            if results.get(key) is None:
                continue
            mt = results[key]["lac_mean"]; st = results[key]["lac_std"]
            c  = COLORS[key]
            cf = _lighten_toward_white(c)
            ax.plot(T_GRID, mt, color=c, lw=2.0, label=LABELS[key])
            ax.fill_between(T_GRID, mt - st, mt + st, color=cf, alpha=0.35)
        ax.axhline(LAC_DANGER_THR, color="#4A148C", ls="--", lw=1.2,
                   label=f"danger ({LAC_DANGER_THR} mmol/L)")
        ax.axhline(LAC_NORMAL_THR, color="darkorange", ls=":", lw=1.0,
                   label=f"normal ({LAC_NORMAL_THR} mmol/L)")
        ax.axvline(POST_HOUR, color="gray", ls="--", lw=1.0)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("Lactate (mmol/L)")
        ax.set_title(f"Lactate — {title}"); ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sofa_pid{pid}_{tag}.png"), dpi=150)
        plt.close(fig)


# ─── Summary plots ───────────────────────────────────────────────────────────

def _plot_summary(all_results: dict, save_dir: str, algo: str, K: int = 20):
    os.makedirs(save_dir, exist_ok=True)
    POLICIES = _policies_for(algo)
    LABELS   = _labels_for(algo)
    COLORS   = _colors_for(algo)

    # File-name prefix used by every plot below: e.g. "trpo_K20_..."
    FNAME_PREFIX = f"{algo}_K{K}_"

    pids = sorted(all_results.keys())

    def _extr(key, metric):
        return np.array([
            all_results[p][key][metric]
            if all_results[p].get(key) is not None else np.nan
            for p in pids
        ])

    metrics_table = {}
    for key in POLICIES:
        metrics_table[key] = {
            "sofa20":   _extr(key, "mean_sofa"),
            "sofa_all": _extr(key, "mean_sofa_all"),
            "sofa96":   _extr(key, "final_sofa"),
            "mean_lac": _extr(key, "mean_lac"),
            "safety":   _extr(key, "safety_rate"),
            "clr6h":    _extr(key, "clearance_6h"),
            "tabove4":  _extr(key, "time_above_4"),
            "mean_dt":  _extr(key, "mean_dt"),
        }

    def _strip(v): return v[~np.isnan(v)]

    def _ylim(metric_name, pad=0.5):
        vals = np.concatenate([_strip(metrics_table[k][metric_name]) for k in POLICIES])
        if len(vals) == 0:
            return (0.0, 1.0)
        return float(np.nanmin(vals)) - pad, float(np.nanmax(vals)) + pad

    ymin_s,  ymax_s  = _ylim("sofa20")
    ymin_sa, ymax_sa = _ylim("sofa_all")
    ymin_sf, ymax_sf = _ylim("sofa96")

    xs = np.arange(len(pids))

    def _ind_wins_lower_is_better(v_ind, v_pop):
        m = np.isfinite(v_ind) & np.isfinite(v_pop)
        if not np.any(m):
            return 0
        return int(np.sum(v_ind[m] < v_pop[m]))

    # Match the dist plot font scale (≈ 2× the original line plots)
    AXIS_LABEL_FS_SOFA = 36
    TICK_FS_SOFA       = 24
    LEGEND_FS_SOFA     = 28

    def _sofa_line(fname, metric, ylabel, ylims, dt_tag):
        ind_key = f"{algo}_ind_{dt_tag}"
        pop_key = f"{algo}_pop_{dt_tag}"
        v_ind = np.asarray(metrics_table[ind_key][metric], dtype=float)
        v_pop = np.asarray(metrics_table[pop_key][metric], dtype=float)
        n_win   = _ind_wins_lower_is_better(v_ind, v_pop)
        n_valid = int(np.sum(np.isfinite(v_ind) & np.isfinite(v_pop)))
        win_pct = n_win / n_valid * 100 if n_valid > 0 else 0.0
        ind_lbl = f"{LABELS[ind_key]}  win:{win_pct:.0f}%"

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(xs, v_ind, color=COLOR_IND, lw=2.2, label=ind_lbl,
                marker="o", ms=8, mfc=COLOR_IND, mec="#333333", mew=1.0, zorder=3)
        ax.plot(xs, v_pop, color=COLOR_POP, lw=2.2, label=LABELS[pop_key],
                marker="s", ms=7, mfc=COLOR_POP, mec="#333333", mew=1.0, zorder=3)
        ax.set_xlabel("Patient (sorted by ID)", fontsize=AXIS_LABEL_FS_SOFA)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS_SOFA)
        ax.set_xticks([])
        ax.legend(fontsize=LEGEND_FS_SOFA, loc="best")
        ax.tick_params(labelsize=TICK_FS_SOFA)
        ax.set_ylim(*ylims)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, FNAME_PREFIX + fname),
                    dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    for dt_tag in ("fixdt", "vardt"):
        _sofa_line(f"mean_sofa_per_patient_{dt_tag}.png",
                   "sofa20", "Mean SOFA (t ≥ 20 h)", (ymin_s, ymax_s), dt_tag)
        _sofa_line(f"mean_sofa_alltime_per_patient_{dt_tag}.png",
                   "sofa_all", "Mean SOFA (all time)", (ymin_sa, ymax_sa), dt_tag)
        _sofa_line(f"final_sofa_per_patient_{dt_tag}.png",
                   "sofa96", "Final SOFA (t = 96 h)", (ymin_sf, ymax_sf), dt_tag)

    # KDE distributions
    try:
        from scipy.stats import gaussian_kde
        kde_ok = True
    except Exception:
        kde_ok = False

    def _sofa_dist(fname, metric, xlabel, xlims, dt_tag):
        if not kde_ok:
            return
        ind_key = f"{algo}_ind_{dt_tag}"
        pop_key = f"{algo}_pop_{dt_tag}"
        v_ind = _strip(np.asarray(metrics_table[ind_key][metric], dtype=float))
        v_pop = _strip(np.asarray(metrics_table[pop_key][metric], dtype=float))
        if len(v_ind) < 2 or len(v_pop) < 2:
            return
        grid = np.linspace(xlims[0], xlims[1], 400)
        try:
            kde_ind = gaussian_kde(v_ind)
            kde_pop = gaussian_kde(v_pop)
        except Exception as e:
            print(f"[KDE] {fname}: skipped ({e})", flush=True)
            return
        d_ind = kde_ind(grid); d_pop = kde_pop(grid)

        _v_ind2 = np.asarray(metrics_table[ind_key][metric], dtype=float)
        _v_pop2 = np.asarray(metrics_table[pop_key][metric], dtype=float)
        n_win   = _ind_wins_lower_is_better(_v_ind2, _v_pop2)
        n_valid = int(np.sum(np.isfinite(_v_ind2) & np.isfinite(_v_pop2)))
        win_pct = n_win / n_valid * 100 if n_valid > 0 else 0.0

        # Larger fonts for distribution plots specifically (≈ 2× line plots)
        AXIS_LABEL_FS_DIST = 36
        TICK_FS_DIST       = 24
        LEGEND_FS_DIST     = 28

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(grid, d_ind, color=COLOR_IND, alpha=0.18)
        ax.plot(grid, d_ind, color=COLOR_IND, lw=2.5,
                label=f"{LABELS[ind_key]}  win:{win_pct:.0f}%")
        ax.axvline(float(np.mean(v_ind)), color=COLOR_IND, ls="--", lw=1.4, alpha=0.85)
        ax.fill_between(grid, d_pop, color=COLOR_POP, alpha=0.18)
        ax.plot(grid, d_pop, color=COLOR_POP, lw=2.5, label=LABELS[pop_key])
        ax.axvline(float(np.mean(v_pop)), color=COLOR_POP, ls="--", lw=1.4, alpha=0.85)
        ax.set_xlabel(xlabel,             fontsize=AXIS_LABEL_FS_DIST)
        ax.set_ylabel("Probability density", fontsize=AXIS_LABEL_FS_DIST)
        ax.set_xlim(*xlims); ax.set_ylim(bottom=0)
        ax.legend(fontsize=LEGEND_FS_DIST, loc="best")
        ax.tick_params(labelsize=TICK_FS_DIST)
        plt.tight_layout()
        # bbox_inches="tight" + extra pad ensures large axis labels are not clipped
        plt.savefig(os.path.join(save_dir, FNAME_PREFIX + fname),
                    dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    for dt_tag in ("fixdt", "vardt"):
        _sofa_dist(f"mean_sofa_dist_{dt_tag}.png",
                   "sofa20", "Mean SOFA (t ≥ 20 h)", (ymin_s, ymax_s), dt_tag)
        _sofa_dist(f"mean_sofa_alltime_dist_{dt_tag}.png",
                   "sofa_all", "Mean SOFA (all time)", (ymin_sa, ymax_sa), dt_tag)
        _sofa_dist(f"final_sofa_dist_{dt_tag}.png",
                   "sofa96", "Final SOFA (t = 96 h)", (ymin_sf, ymax_sf), dt_tag)

    # Safety rate per-patient
    def _sr_line(fname, dt_tag):
        ind_key = f"{algo}_ind_{dt_tag}"
        pop_key = f"{algo}_pop_{dt_tag}"
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(xs, metrics_table[ind_key]["safety"] * 100,
                color=COLOR_IND, lw=2.2, marker="o", ms=8, mec="#333333", mew=1.0,
                label=LABELS[ind_key])
        ax.plot(xs, metrics_table[pop_key]["safety"] * 100,
                color=COLOR_POP, lw=2.2, marker="s", ms=7, mec="#333333", mew=1.0,
                label=LABELS[pop_key])
        ax.set_xlabel("Patient (sorted by ID)", fontsize=AXIS_LABEL_FS_SOFA)
        ax.set_ylabel(f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]",
                      fontsize=AXIS_LABEL_FS_SOFA)
        ax.set_xticks([])
        ax.set_ylim(0, 105)
        ax.legend(fontsize=LEGEND_FS_SOFA)
        ax.tick_params(labelsize=TICK_FS_SOFA)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, FNAME_PREFIX + fname),
                    dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    for dt_tag in ("fixdt", "vardt"):
        _sr_line(f"safety_rate_{dt_tag}.png", dt_tag)

    # 4-policy boxplots
    order4  = POLICIES
    labels4 = [LABELS[k] for k in order4]
    colors4 = [COLORS[k] for k in order4]

    def _make_boxplot(fname, metric, ylabel, ylim, title, scale=1.0):
        fig, ax = plt.subplots(figsize=(11, 6))
        data = [_strip(metrics_table[k][metric]) * scale for k in order4]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels4)
        for patch, c, k in zip(bp["boxes"], colors4, order4):
            patch.set_facecolor(c); patch.set_alpha(0.72)
            patch.set_edgecolor("0.35"); patch.set_linewidth(0.9)
            if k.endswith("vardt"):
                patch.set_hatch("//")
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS_SOFA)
        ax.tick_params(labelsize=TICK_FS_SOFA)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, FNAME_PREFIX + fname),
                    dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    _make_boxplot("boxplot_sofa_post20h.png", "sofa20",
                  "Mean SOFA (t ≥ 20 h)", (ymin_s, ymax_s), "SOFA (t ≥ 20 h)")
    _make_boxplot("boxplot_sofa_alltime.png", "sofa_all",
                  "Mean SOFA (all time)", (ymin_sa, ymax_sa), "SOFA (all time)")
    _make_boxplot("boxplot_sofa_final.png", "sofa96",
                  "Final SOFA (t = 96 h)", (ymin_sf, ymax_sf),
                  "Final SOFA (t = 96 h)")
    _make_boxplot("boxplot_safety_rate.png", "safety",
                  f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]", (0, 105),
                  "Lactate Safety Rate (all time)", scale=100.0)

    # Overall stats bar chart
    sm_post = [np.nanmean(metrics_table[k]["sofa20"])    for k in order4]
    ss_post = [np.nanstd (metrics_table[k]["sofa20"])    for k in order4]
    sm_all  = [np.nanmean(metrics_table[k]["sofa_all"])  for k in order4]
    ss_all  = [np.nanstd (metrics_table[k]["sofa_all"])  for k in order4]
    sm_fin  = [np.nanmean(metrics_table[k]["sofa96"])    for k in order4]
    ss_fin  = [np.nanstd (metrics_table[k]["sofa96"])    for k in order4]
    sr_m    = [np.nanmean(metrics_table[k]["safety"]) * 100 for k in order4]
    sr_s    = [np.nanstd (metrics_table[k]["safety"]) * 100 for k in order4]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    xs_bar = np.arange(len(labels4))
    for ax, ymeans, ystds, ylabel, title in [
            (axes[0], sm_post, ss_post, "Mean SOFA (t ≥ 20 h)",  "SOFA (t ≥ 20 h)"),
            (axes[1], sm_all,  ss_all,  "Mean SOFA (all time)",  "SOFA (all time)"),
            (axes[2], sm_fin,  ss_fin,  "Final SOFA (t = 96 h)", "Final SOFA"),
            (axes[3], sr_m,    sr_s,
             f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]",
             "Lactate Safety Rate"),
    ]:
        ax.bar(xs_bar, ymeans, yerr=ystds, color=colors4, alpha=0.8,
               capsize=5, error_kw={"lw": 1.5})
        ax.set_xticks(xs_bar)
        ax.set_xticklabels(labels4, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=11)
    axes[3].set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, FNAME_PREFIX + "overall_stats.png"),
                dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    # Print + save text summary
    lines = [
        "policy,n,sofa_post20h_mean,sofa_post20h_std,sofa_alltime_mean,sofa_alltime_std,"
        "final_sofa_mean,final_sofa_std,safety_rate_mean,safety_rate_std,"
        "mean_lac_mean,mean_lac_std,clearance_6h_mean,time_above_4_mean,mean_dt"
    ]
    print("\n" + "=" * 110)
    print(f"{'Policy':<22} {'N':>4} "
          f"{'SOFA(≥20h)':>11} {'±std':>7}  "
          f"{'SOFA(all)':>10} {'±std':>7}  "
          f"{'SOFA(96h)':>10} {'±std':>7}  "
          f"{'Safe%':>7} {'±std':>6}  "
          f"{'lac':>5}  {'dt':>5}")
    print("-" * 110)
    for key in order4:
        sofa20   = metrics_table[key]["sofa20"]
        sofa_all = metrics_table[key]["sofa_all"]
        sofa96   = metrics_table[key]["sofa96"]
        safety   = metrics_table[key]["safety"]
        mean_lac = metrics_table[key]["mean_lac"]
        clr6h    = metrics_table[key]["clr6h"]
        tabove4  = metrics_table[key]["tabove4"]
        mean_dt  = metrics_table[key]["mean_dt"]
        n = int(np.sum(~np.isnan(sofa20)))
        if n == 0:
            print(f"{key:<22} {n:>4}  (no data)")
            lines.append(f"{key},{n},,,,,,,,,,,,")
            continue
        print(
            f"{key:<22} {n:>4} "
            f"{np.nanmean(sofa20):>11.3f} {np.nanstd(sofa20):>7.3f}  "
            f"{np.nanmean(sofa_all):>10.3f} {np.nanstd(sofa_all):>7.3f}  "
            f"{np.nanmean(sofa96):>10.3f} {np.nanstd(sofa96):>7.3f}  "
            f"{np.nanmean(safety) * 100:>6.1f}% {np.nanstd(safety) * 100:>5.1f}  "
            f"{np.nanmean(mean_lac):>5.2f}  {np.nanmean(mean_dt):>5.2f}"
        )
        lines.append(
            f"{key},{n},"
            f"{np.nanmean(sofa20):.4f},{np.nanstd(sofa20):.4f},"
            f"{np.nanmean(sofa_all):.4f},{np.nanstd(sofa_all):.4f},"
            f"{np.nanmean(sofa96):.4f},{np.nanstd(sofa96):.4f},"
            f"{np.nanmean(safety):.4f},{np.nanstd(safety):.4f},"
            f"{np.nanmean(mean_lac):.4f},{np.nanstd(mean_lac):.4f},"
            f"{np.nanmean(clr6h):.4f},{np.nanmean(tabove4):.4f},{np.nanmean(mean_dt):.4f}"
        )
    print("=" * 110 + "\n")

    with open(os.path.join(save_dir, "overall_stats.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    np.save(os.path.join(save_dir, "summary_scores.npy"), {
        "algo": algo,
        "pids": pids,
        **{f"{key}_{m}": metrics_table[key][m]
           for key in POLICIES
           for m in ["sofa20", "sofa_all", "sofa96", "mean_lac",
                     "safety", "clr6h", "tabove4", "mean_dt"]},
    })

    pp_lines = ["pid,policy,sofa20,sofa_all,sofa96,mean_lac,safety,clearance_6h,"
                "time_above_4,mean_dt"]
    for i, pid in enumerate(pids):
        for key in POLICIES:
            mt = metrics_table[key]
            pp_lines.append(
                f"{pid},{key},"
                f"{mt['sofa20'][i]:.4f},{mt['sofa_all'][i]:.4f},"
                f"{mt['sofa96'][i]:.4f},{mt['mean_lac'][i]:.4f},"
                f"{mt['safety'][i]:.4f},{mt['clr6h'][i]:.4f},"
                f"{mt['tabove4'][i]:.4f},{mt['mean_dt'][i]:.4f}"
            )
    with open(os.path.join(save_dir, "per_patient_metrics.csv"), "w") as f:
        f.write("\n".join(pp_lines) + "\n")

    # ── Improvement (% change) CSVs ─────────────────────────────────────
    # SOFA is "lower is better", so a *negative* change rate means improvement.
    #   pop2ind_<dt>  : (sofa_ind  - sofa_pop ) / sofa_pop
    #   fix2var_<who> : (sofa_vardt - sofa_fixdt) / sofa_fixdt
    # Saved to per_patient_improvement.csv (raw, per patient) and
    # improvement_stats.csv (mean ± std/var across patients).
    SOFA_METRICS = [("sofa20", "SOFA(>=20h)"),
                    ("sofa_all", "SOFA(all)"),
                    ("sofa96", "SOFA(96h)")]

    def _safe_rate(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (num - den) / den
        r = np.where(np.isfinite(r), r, np.nan)
        return r

    # Build wide-format per-patient matrix
    imp_cols = ["pid"]
    imp_data: dict[str, np.ndarray] = {}
    for m_key, _ in SOFA_METRICS:
        for tag in ("fixdt", "vardt"):
            col = f"{m_key}_pop2ind_{tag}"
            imp_cols.append(col)
            imp_data[col] = _safe_rate(
                np.asarray(metrics_table[f"{algo}_ind_{tag}"][m_key], dtype=float),
                np.asarray(metrics_table[f"{algo}_pop_{tag}"][m_key], dtype=float),
            )
    for m_key, _ in SOFA_METRICS:
        for which in ("ind", "pop"):
            col = f"{m_key}_fix2var_{which}"
            imp_cols.append(col)
            imp_data[col] = _safe_rate(
                np.asarray(metrics_table[f"{algo}_{which}_vardt"][m_key], dtype=float),
                np.asarray(metrics_table[f"{algo}_{which}_fixdt"][m_key], dtype=float),
            )

    pp_imp_lines = [",".join(imp_cols)]
    for i, pid in enumerate(pids):
        row = [str(pid)]
        for col in imp_cols[1:]:
            v = imp_data[col][i]
            row.append(f"{v:.6f}" if np.isfinite(v) else "")
        pp_imp_lines.append(",".join(row))
    with open(os.path.join(save_dir, "per_patient_improvement.csv"), "w") as f:
        f.write("\n".join(pp_imp_lines) + "\n")

    # Aggregate stats across patients
    stat_lines = ["metric,comparison,mean,std,var,n"]
    for col in imp_cols[1:]:
        # col format: "<m_key>_<kind>_<tag_or_who>"
        m_key, kind, tail = col.split("_", 2)
        # Re-stitch m_key for "sofa_all"/"sofa20"/"sofa96"
        if m_key == "sofa" and tail.split("_", 1)[0] in ("all", "20"):
            # cases like "sofa_all_pop2ind_fixdt": already split incorrectly above
            pass
        # Simpler: parse from imp_data definition tables instead
        v = imp_data[col]
        v = v[np.isfinite(v)]
        if v.size == 0:
            stat_lines.append(f"{col},,nan,nan,nan,0")
            continue
        m  = float(v.mean())
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        va = float(v.var(ddof=1)) if v.size > 1 else 0.0
        # Re-derive metric / comparison from col name reliably
        if "_pop2ind_" in col:
            metric_part, _, dt_part = col.partition("_pop2ind_")
            comparison = f"pop2ind_{dt_part}"
        elif "_fix2var_" in col:
            metric_part, _, who_part = col.partition("_fix2var_")
            comparison = f"fix2var_{who_part}"
        else:
            metric_part, comparison = col, ""
        stat_lines.append(f"{metric_part},{comparison},{m:.6f},{sd:.6f},{va:.6f},{v.size}")
    with open(os.path.join(save_dir, "improvement_stats.csv"), "w") as f:
        f.write("\n".join(stat_lines) + "\n")


def _rebuild_all_results_from_summary_npy(path: str, algo: str) -> dict:
    """Rebuild minimal all_results dict for _plot_summary from summary_scores.npy."""
    raw = np.load(path, allow_pickle=True)
    blob = raw.item() if getattr(raw, "shape", None) == () else raw
    if "algo" in blob and blob["algo"] != algo:
        print(f"[plot_only] WARNING: --algo={algo} but summary_scores.npy "
              f"was written for algo={blob['algo']!r}", flush=True)
    pids_order = [int(x) for x in blob["pids"]]
    POLICIES = _policies_for(algo)
    METRIC_KEYS = [
        ("sofa20",   "mean_sofa"),
        ("sofa_all", "mean_sofa_all"),
        ("sofa96",   "final_sofa"),
        ("mean_lac", "mean_lac"),
        ("safety",   "safety_rate"),
        ("clr6h",    "clearance_6h"),
        ("tabove4",  "time_above_4"),
        ("mean_dt",  "mean_dt"),
    ]
    all_results: dict = {}
    for idx, pid in enumerate(pids_order):
        all_results[pid] = {}
        for policy in POLICIES:
            s20_key = f"{policy}_sofa20"
            if s20_key not in blob:
                all_results[pid][policy] = None
                continue
            s20 = np.asarray(blob[s20_key], dtype=float)
            val = float(s20.flat[idx]) if s20.size > idx else float("nan")
            if np.isnan(val):
                all_results[pid][policy] = None
                continue
            entry = {}
            for mk, rk in METRIC_KEYS:
                a = np.asarray(blob[f"{policy}_{mk}"], dtype=float)
                entry[rk] = float(a.flat[idx]) if a.size > idx else float("nan")
            all_results[pid][policy] = entry
    return all_results


# ─── Patient discovery ───────────────────────────────────────────────────────

def _discover_default_patients(ind_root: Path, pinn_root: Path,
                               file_prefix: str, K: int) -> list[int]:
    pids: list[int] = []
    if not ind_root.exists():
        return pids
    for d in sorted(ind_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("patient_"):
            continue
        try:
            pid = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        if _resolve_pinn_path(pinn_root, pid) is None:
            continue
        # Patient is eligible if at least one ind ckpt exists.
        if any(
            _resolve_ind_path(ind_root, pid, tag, K, file_prefix) is not None
            for tag in ("fixdt", "vardt")
        ):
            pids.append(pid)
    return pids


# ─── Main ────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Compare per-patient (Ind) vs population (Pop) policies "
                    "for a chosen algorithm on per-patient PINNs."
    )
    p.add_argument("--algo", required=True, choices=sorted(ALGO_REG.keys()),
                   help="Algorithm to evaluate.")
    p.add_argument("--pinn_dir",    default=PINN_IND_ROOT,
                   help="Per-patient PINN root.")
    p.add_argument("--ind_root",    default=POLICY_IND_ROOT,
                   help="Per-patient (Ind) weight root. "
                        "Defaults to the algorithm-specific path.")
    p.add_argument("--pop_root",    default=POLICY_POP_ROOT,
                   help="Population (Pop) weight root: "
                        "<pop_root>/<algo>/K<K>/<tag>/policy.pt (legacy paths "
                        "also tried).")
    p.add_argument(
        "--glb_scales",
        default=str(Path(PINN_POP_ROOT) / "scales.npy"),
        help="Global scales.npy used to train population policies.",
    )
    p.add_argument(
        "--save_root",
        default=str(POLICY_TRANSFER_EVAL_ROOT),
        help="Root directory for transfer evaluation outputs.",
    )
    p.add_argument(
        "--save_dir",
        default=None,
        help="Optional explicit output directory. If omitted, use "
            "<save_root>/<algo>/K<K>/n<n_eval>_noise<init_noise_std>/.",
    )
    p.add_argument("--n_eval",      type=int,   default=50)
    p.add_argument("--K",           type=int,   default=20)
    p.add_argument("--init_noise_std",  type=float, default=0.1,
                   help="Std of Gaussian noise on init_state_norm (PINN norm space). "
                        "Use 0.0 for deterministic init from init_state_norm.npy.")
    p.add_argument("--patient_ids", type=int, nargs="*", default=None,
                   help="Override patient ID list "
                        "(default: all under ind_root that have a PINN).")
    p.add_argument("--plot_only", action="store_true",
                   help="Re-render figures from summary/summary_scores.npy "
                        "without re-running rollouts.")
    return p.parse_args()


def main():
    def _noise_tag(x: float) -> str:
        return f"{x:g}".replace(".", "p")
    args = _parse()
    algo = args.algo
    cfg  = ALGO_REG[algo]

    if args.ind_root is None:
        args.ind_root = cfg["default_ind_root"]
    if args.save_dir is None:
        args.save_dir = (
            Path(args.save_root)
            / algo
            / f"K{args.K}"
            / f"n{args.n_eval}_noise{_noise_tag(args.init_noise_std)}"
        )
    else:
        args.save_dir = Path(args.save_dir)

    summary_dir = os.path.join(args.save_dir, "summary")

    if args.plot_only:
        npy = os.path.join(summary_dir, "summary_scores.npy")
        if not os.path.isfile(npy):
            raise SystemExit(
                f"--plot_only: file not found: {npy}\n"
                "Run once without --plot_only (same --save_dir) to create it."
            )
        os.makedirs(summary_dir, exist_ok=True)
        all_results = _rebuild_all_results_from_summary_npy(npy, algo)
        print(f"[plot_only] {len(all_results)} patients ← {npy}")
        _plot_summary(all_results, summary_dir, algo, args.K)
        print(f"[plot_only] updated → {summary_dir}")
        return

    pinn_root = Path(args.pinn_dir)
    if not pinn_root.exists():
        raise SystemExit(f"--pinn_dir not found: {pinn_root}")
    ind_root = Path(args.ind_root)
    if not ind_root.exists():
        raise SystemExit(f"--ind_root not found: {ind_root}")

    if args.patient_ids:
        pids = list(args.patient_ids)
    else:
        pids = _discover_default_patients(ind_root, pinn_root,
                                          cfg["file_prefix"], args.K)
    if not pids:
        raise SystemExit("No patients to evaluate.")

    per_patient_dir = os.path.join(args.save_dir, "per_patient")
    os.makedirs(per_patient_dir, exist_ok=True)
    os.makedirs(summary_dir,     exist_ok=True)
    _save_meta(args, pids, Path(args.save_dir))

    print(f"\n{'='*72}")
    print(f"  Transfer evaluation: {cfg['label']} Ind vs Pop")
    print(f"  algo      : {algo}")
    print(f"  pinn_dir  : {args.pinn_dir}")
    print(f"  ind_root  : {args.ind_root}")
    print(f"  pop_root  : {args.pop_root}")
    print(f"  glb_scales: {args.glb_scales}")
    print(f"  save_dir  : {args.save_dir}")
    print(f"  n_eval    : {args.n_eval}  K={args.K}  init_noise_std={args.init_noise_std}")
    print(f"  patients  : {len(pids)}  (first 5: {pids[:5]}{' ...' if len(pids) > 5 else ''})")
    print(f"  state_min/max: physical ∩ data (patient PINN env)")
    print(f"  pop policies wrapped in AdaptedEnv (state + action scale conversion)")
    print(f"{'='*72}\n")

    all_results = {}
    for pid in pids:
        res = _worker(pid, algo, args.pinn_dir, args.ind_root, args.pop_root,
                      args.glb_scales,
                      args.n_eval, args.K,
                      init_noise_std=args.init_noise_std)
        if res is None:
            continue
        all_results[pid] = res
        _plot_per_patient(pid, res, per_patient_dir, algo)
        _save_trajectories(pid, res, per_patient_dir, algo)

    if not all_results:
        raise SystemExit("No results collected.")

    _plot_summary(all_results, summary_dir, algo, args.K)
    print(f"\nDone.  Plots & trajectories saved under {args.save_dir}")


if __name__ == "__main__":
    main()
