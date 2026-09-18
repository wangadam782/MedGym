"""collect_offline_data.py — Collect offline behavior datasets from online policies.

This script rolls out individual and population Lagrangian TRPO behavior
policies on individual patient PINN simulators and saves CQL-compatible NPZ
datasets under `results/offline/datasets/`.

No-MIMIC workflow:
  Use --no_mimic_csv with released patient PINN folders. Initial states are
  loaded from each patient's `init_state_norm.npy`; the MIMIC-derived CSV is
  only used as a development fallback when that file is unavailable.

Patient selection priority:
  1. --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv
  2. --patient_ids 200325 201046 ...
  3. all discoverable patient_<pid>/ folders under --pinn_dir

Saved datasets:
  <save_dir>/datasets/pid_<pid>_fixdt_dataset.npz
  <save_dir>/datasets/trpo_pop_fixdt_dataset.npz

Usage:
    python3 scripts/offline/collect_offline_data.py \
        --no_mimic_csv \
        --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv \
        --pinn_dir results/pinn/individual \
        --ind_root results/online/individual \
        --glb_root results/online/population \
        --glb_scales results/pinn/population/scales.npy \
        --save_dir results/offline \
        --behavior_algo lagrangian_trpo \
        --dt_mode fixdt \
        --n_eval 3 \
        --K 20
"""
import argparse
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from data.config import state_dim, action_dim
from data.loader import load_all_patients

POST_HOUR        = 20.0
TOTAL_TIME_H     = 96.0
T_GRID           = np.arange(0, TOTAL_TIME_H + 1, 1.0)
LAC_DANGER_THR   = 4.0   # mmol/L — septic shock threshold
LAC_NORMAL_THR   = 2.0   # mmol/L — normal range

POLICIES = ["ind_fixdt", "ind_vardt", "glb_fixdt", "glb_vardt"]
LABELS = {
    "ind_fixdt": "ind Lagrangian-TRPO (fixed-dt)",
    "ind_vardt": "ind Lagrangian-TRPO (var-dt)",
    "glb_fixdt": "global Lagrangian-TRPO (fixed-dt)",
    "glb_vardt": "global Lagrangian-TRPO (var-dt)",
}
COLORS = {
    "ind_fixdt": "#1f77b4",
    "ind_vardt": "#aec7e8",
    "glb_fixdt": "#d62728",
    "glb_vardt": "#ff9896",
}


# ─── AdaptedEnv ──────────────────────────────────────────────────────────────

class AdaptedEnv:
    """Wrap a per-patient environment for evaluation with a globally-trained agent.

    Converts SMDP states from per-patient normalisation to global normalisation.
    """
    def __init__(self, patient_env, mean_glob, std_glob, mean_pat, std_pat):
        self._env       = patient_env
        self.mean_glob  = mean_glob
        self.std_glob   = std_glob
        self.mean_pat   = mean_pat
        self.std_pat    = std_pat
        self.smdp_state_dim    = patient_env.smdp_state_dim
        self.option_action_dim = patient_env.option_action_dim
        self.dt_min            = patient_env.dt_min
        self.dt_max            = patient_env.dt_max
        self.total_time_h      = patient_env.total_time_h

    def _adapt_state(self, state: np.ndarray) -> np.ndarray:
        phys_pat  = state[:state_dim]
        phys_real = phys_pat * self.std_pat[:state_dim] + self.mean_pat[:state_dim]
        phys_glob = (phys_real - self.mean_glob[:state_dim]) / self.std_glob[:state_dim]
        return np.concatenate([phys_glob, state[state_dim:]], dtype=np.float32)

    def reset(self, **kwargs):
        return self._adapt_state(self._env.reset(**kwargs))

    def step(self, action):
        next_state, reward, done, info = self._env.step(action)
        return self._adapt_state(next_state), reward, done, info

    @property
    def current_t(self):
        return self._env.current_t


# ─── Rollout ─────────────────────────────────────────────────────────────────

def _interp_zoh(t_arr, val_arr, t_grid):
    """Zero-order hold interpolation onto t_grid."""
    out = np.empty(len(t_grid), dtype=np.float32)
    for gi, tg in enumerate(t_grid):
        idx = max(0, int(np.searchsorted(t_arr, tg, side="right")) - 1)
        out[gi] = val_arr[min(idx, len(val_arr) - 1)]
    return out


def _run_episode_traj(agent, env, init_state_norm, action_noise_std: float = 0.0,
                      rng: np.random.Generator = None):
    """Run one episode; return (sofa_interp, lac_interp, metrics_dict, transitions).

action_noise_std > 0 adds Gaussian noise to each action before
    env.step(), clipped to [-1, 1].  This increases dataset diversity for CQL
    without meaningfully changing evaluation metrics when noise is small.

Returns a 4th element `transitions`: a dict of lists containing
    raw (s, a, r, ns, done, sofa, lactate) for each step, using only the
    physical state dims (first state_dim entries, t/k stripped).
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(init_state_norm, dtype=np.float32)
    state = env.reset(init_state_norm=x0)
    done  = False
    t_list, sofa_list, lac_list = [], [], []


    trans_s, trans_a, trans_r, trans_ns, trans_done = [], [], [], [], []
    trans_sofa, trans_lac = [], []

    while not done:
        t0     = float(env.current_t)
        action = agent.select_action(state, deterministic=True)


        if action_noise_std > 0.0:
            noise  = rng.normal(0.0, action_noise_std, size=action.shape).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)

        next_state, reward, done, info = env.step(action)

        t_list.append(t0)
        sofa_list.append(float(info["sofa"]))
        lac_list.append(float(info["lactate"]))


        trans_s.append(state[:state_dim].copy())
        trans_a.append(action[:3].copy())
        trans_r.append(float(reward))
        trans_ns.append(next_state[:state_dim].copy())
        trans_done.append(float(done))
        trans_sofa.append(float(info["sofa"]))
        trans_lac.append(float(info["lactate"]))

        state = next_state

    t_arr    = np.array(t_list,    dtype=np.float32)
    sofa_arr = np.array(sofa_list, dtype=np.float32)
    lac_arr  = np.array(lac_list,  dtype=np.float32)

    sofa_interp = _interp_zoh(t_arr, sofa_arr, T_GRID)
    lac_interp  = _interp_zoh(t_arr, lac_arr,  T_GRID)

    post_mask     = T_GRID >= POST_HOUR
    mean_sofa     = float(sofa_interp[post_mask].mean())
    mean_sofa_all = float(sofa_interp.mean())

    mean_lac    = float(lac_interp.mean())
    safety_rate = float(np.mean(lac_interp < LAC_DANGER_THR))

    lac_t0 = float(lac_interp[0])
    t6_idx = int(np.searchsorted(T_GRID, 6.0))
    lac_t6 = float(lac_interp[min(t6_idx, len(lac_interp) - 1)])
    clearance_6h = float((lac_t0 - lac_t6) / lac_t0) if lac_t0 > 0 else 0.0

    time_above_4 = float(np.sum(lac_interp > LAC_DANGER_THR))

    metrics = {
        "mean_sofa":     mean_sofa,
        "mean_sofa_all": mean_sofa_all,
        "mean_lac":      mean_lac,
        "safety_rate":   safety_rate,
        "clearance_6h":  clearance_6h,
        "time_above_4":  time_above_4,
    }


    transitions = {
        "s":       np.array(trans_s,    dtype=np.float32),
        "a":       np.array(trans_a,    dtype=np.float32),
        "r":       np.array(trans_r,    dtype=np.float32),
        "ns":      np.array(trans_ns,   dtype=np.float32),
        "done":    np.array(trans_done, dtype=np.float32),
        "sofa":    np.array(trans_sofa, dtype=np.float32),
        "lactate": np.array(trans_lac,  dtype=np.float32),
    }

    return sofa_interp, lac_interp, metrics, transitions


def _collect_policy(agent, env, init_state_norm, n_eval: int,
                    action_noise_std: float = 0.0,
                    rng: np.random.Generator = None):
    """Run n_eval episodes; aggregate trajectories, safety metrics, and transitions.

Passes action_noise_std / rng through to _run_episode_traj.
Returns a 6th element: the combined offline dataset dict ready
    for np.savez_compressed().
    """
    if rng is None:
        rng = np.random.default_rng()

    sofa_trajs, lac_trajs = [], []
    all_metrics = {k: [] for k in
                   ["mean_sofa", "mean_sofa_all", "mean_lac",
                    "safety_rate", "clearance_6h", "time_above_4"]}


    all_s, all_a, all_r, all_ns, all_done = [], [], [], [], []
    all_sofa_steps, all_lac_steps, all_ep_id = [], [], []
    ep_lengths, ep_reward_sum = [], []

    for ep_idx in range(n_eval):
        sofa_t, lac_t, m, trans = _run_episode_traj(
            agent, env, init_state_norm,
            action_noise_std=action_noise_std,
            rng=rng,
        )
        sofa_trajs.append(sofa_t)
        lac_trajs.append(lac_t)
        for k in all_metrics:
            all_metrics[k].append(m[k])


        n_steps = len(trans["r"])
        all_s.append(trans["s"])
        all_a.append(trans["a"])
        all_r.append(trans["r"])
        all_ns.append(trans["ns"])
        all_done.append(trans["done"])
        all_sofa_steps.append(trans["sofa"])
        all_lac_steps.append(trans["lactate"])
        all_ep_id.append(np.full(n_steps, ep_idx, dtype=np.int32))
        ep_lengths.append(n_steps)
        ep_reward_sum.append(float(trans["r"].sum()))

    sofa_mat = np.stack(sofa_trajs, axis=0)
    lac_mat  = np.stack(lac_trajs,  axis=0)


    def _cat(lst): return np.concatenate(lst, axis=0)

    dataset = {
        "transitions/s":      _cat(all_s),
        "transitions/a":      _cat(all_a),
        "transitions/r":      _cat(all_r),
        "transitions/ns":     _cat(all_ns),
        "transitions/done":   _cat(all_done),
        "meta/sofa":          _cat(all_sofa_steps),
        "meta/lactate":       _cat(all_lac_steps),
        "meta/ep_id":         _cat(all_ep_id),
        "meta/ep_length":     np.array(ep_lengths,    dtype=np.int32),
        "meta/ep_reward_sum": np.array(ep_reward_sum, dtype=np.float32),
        "info/n_eval":        np.array([n_eval]),
        "info/action_noise_std": np.array([action_noise_std]),
        "info/state_dim":     np.array([state_dim]),
        "info/action_dim":    np.array([all_a[0].shape[-1]]),
        "info/n_transitions": np.array([len(_cat(all_r))]),
    }

    return (sofa_mat.mean(0), sofa_mat.std(0),
            lac_mat.mean(0),  lac_mat.std(0),
            {k: float(np.mean(v)) for k, v in all_metrics.items()},
            dataset)


# ─── Dataset saving helper ────────────────────────────────────────────────────

def _save_dataset(path: str, dataset: dict, pid=None, policy_tag: str = "",
                  action_noise_std: float = 0.0):
    """Save a CQL-compatible offline dataset to a compressed NPZ file.

New helper.  Adds info fields and prints a brief summary.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if pid is not None:
        dataset["info/patient_id"] = np.array([pid])
    dataset["info/policy_tag"]  = np.array([policy_tag])
    np.savez_compressed(path, **dataset)
    n = int(dataset["info/n_transitions"][0])
    print(f"  [NPZ saved] {path}  ({n:,} transitions)", flush=True)


# ─── Per-patient worker ───────────────────────────────────────────────────────
import torch
from models import PINN
from rl import ICUEnvironment, LagrangianTRPO, SMDP_STATE_DIM, OPTION_ACTION_DIM
from data.config import device as global_device

def _worker(pid, pinn_dir, glb_scales_path, ind_root, glb_root, save_dir,
            n_eval, K, device_str, csv_path,
            ind_hidden=128, glb_hidden=128,
            action_noise_std: float = 0.0,
            dataset_dir: str = None,
            behavior_algo: str = "lagrangian_trpo",
            dt_mode: str = "fixdt"):

    device = global_device

    rng = np.random.default_rng()

    # ── Load global scales ────────────────────────────────────────────────
    sc_glob   = np.load(glb_scales_path, allow_pickle=True).item()
    mean_glob = sc_glob["mean"].astype(np.float32)
    std_glob  = sc_glob["std"].astype(np.float32)

    from pinn_bundle import load_init_and_scales_for_patient

    try:
        init_norm, pinn_path, mean_pat, std_pat, asc_pat, state_min_np, state_max_np = (
            load_init_and_scales_for_patient(Path(pinn_dir), pid, csv_path)
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[patient {pid}] {e}, skipping.", flush=True)
        return None

    pinn      = PINN(state_dim, action_dim).to(device)
    pinn.load_state_dict(torch.load(str(pinn_path), map_location=device))
    pinn.eval()

    fixed_dt = TOTAL_TIME_H / K

    def make_patient_env(use_dt):
        return ICUEnvironment(
            pinn_model      = pinn,
            mean_np         = mean_pat,
            std_np          = std_pat,
            action_min_norm = np.zeros(action_dim, dtype=np.float32),
            action_max_norm = np.ones(action_dim,  dtype=np.float32),
            state_min       = state_min_np,
            state_max       = state_max_np,
            action_scale_np = asc_pat,
            max_steps       = K,
            dt_min          = 0.5,
            dt_max          = 36.0,
            total_time_h    = TOTAL_TIME_H,
            use_dt          = use_dt,
            fixed_dt        = fixed_dt,
            use_lac_penalty = True,
        )

    def make_agent(hidden=128):
        return LagrangianTRPO(
            state_dim   = SMDP_STATE_DIM,
            action_dim  = OPTION_ACTION_DIM,
            hidden      = hidden,
        )

    results = {}

    # aggregate them across patients into trpo_pop_*.npz
    glb_datasets = {}

    def _run_and_store(key, agent, env):
        sofa_mean, sofa_std, lac_mean, lac_std, m, dataset = _collect_policy(
            agent, env, init_norm, n_eval,
            action_noise_std=action_noise_std,
            rng=rng,
        )
        results[key] = {
            "sofa_mean": sofa_mean, "sofa_std": sofa_std,
            "lac_mean":  lac_mean,  "lac_std":  lac_std,
            **m,
            "_dataset": dataset,
        }
        print(
            f"[patient {pid}] {key}  "
            f"sofa={m['mean_sofa']:.2f}  "
            f"lac={m['mean_lac']:.2f}  "
            f"safety_rate={m['safety_rate']:.2%}  "
            f"clr6h={m['clearance_6h']:.1%}  "
            f"t>4h={m['time_above_4']:.1f}h",
            flush=True,
        )

    def _load_actor(pt_path):
        sd = torch.load(str(pt_path), map_location=device)
        return sd["actor"] if isinstance(sd, dict) and "actor" in sd else sd

    # ── Per-patient ind policies ──────────────────────────────────────────
    mode_pairs = []
    if dt_mode in ["fixdt", "both"]:
        mode_pairs.append(("fixdt", False))
    if dt_mode in ["vardt", "both"]:
        mode_pairs.append(("vardt", True))
    for tag, use_dt in mode_pairs:
        key     = f"ind_{tag}"
        # pt_path = Path(ind_root) / f"patient_{pid}" / f"{tag}" / f"lagrangian_trpo_{tag}.pt"
        # pt_dir = Path(ind_root) / f"patient_{pid}" / args.behavior_algo / f"K{args.K}" / args.dt_mode
        pt_dir = Path(ind_root) / f"patient_{pid}" / behavior_algo / f"K{K}" / tag
        policy_candidates = [
            pt_dir / "actor.pt",
            pt_dir / "policy.pt",
            pt_dir / f"{behavior_algo}_{tag}.pt",
            pt_dir / f"lagrangian_trpo_{tag}.pt",
        ]
        pt_path = next((p for p in policy_candidates if p.exists()), None)
        if pt_path is None:
            raise FileNotFoundError(f"No policy checkpoint found in {pt_dir}")
        
        if not pt_path.exists():
            print(f"[patient {pid}] {key}: model not found ({pt_path})", flush=True)
            results[key] = None
            continue
        agent = make_agent(hidden=ind_hidden)
        agent.actor.load_state_dict(_load_actor(pt_path))
        _run_and_store(key, agent, make_patient_env(use_dt))


        if results[key] is not None and dataset_dir is not None:
            npz_path = os.path.join(dataset_dir, f"pid_{pid}_{tag}_dataset.npz")
            _save_dataset(
                npz_path,
                dict(results[key]["_dataset"]),   # copy to avoid mutation
                pid=pid,
                policy_tag=f"ind_lagrangian_trpo_{tag}",
                action_noise_std=action_noise_std,
            )

    # ── Global Lagrangian TRPO policies (AdaptedEnv) ──────────────────────
    mode_pairs = []
    if dt_mode in ["fixdt", "both"]:
        mode_pairs.append(("fixdt", False))
    if dt_mode in ["vardt", "both"]:
        mode_pairs.append(("vardt", True))
    for tag, use_dt in mode_pairs:
        key     = f"glb_{tag}"
        pt_dir = Path(glb_root) / behavior_algo / f"K{K}" / tag
        policy_candidates = [
            pt_dir / "actor.pt",
            pt_dir / "policy.pt",
            pt_dir / f"{behavior_algo}_{tag}.pt",
            pt_dir / f"lagrangian_trpo_{tag}.pt",
        ]
        pt_path = next((p for p in policy_candidates if p.exists()), None)

        if pt_path is None:
            print(f"[patient {pid}] {key}: model not found in {pt_dir}", flush=True)
            results[key] = None
            continue
        if not pt_path.exists():
            print(f"[patient {pid}] {key}: model not found ({pt_path})", flush=True)
            results[key] = None
            continue
        agent = make_agent(hidden=glb_hidden)
        agent.actor.load_state_dict(_load_actor(pt_path))
        pat_env = make_patient_env(use_dt)
        _run_and_store(key, agent, AdaptedEnv(pat_env, mean_glob, std_glob, mean_pat, std_pat))


        if results[key] is not None:
            glb_datasets[tag] = results[key]["_dataset"]

    return results, glb_datasets


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _requested_mode_pairs(dt_mode: str):
    """Return [(tag, display title)] for the requested fixed/variable dt modes."""
    pairs = []
    if dt_mode in ("fixdt", "both"):
        pairs.append(("fixdt", "fixed-dt"))
    if dt_mode in ("vardt", "both"):
        pairs.append(("vardt", "variable-dt"))
    return pairs


def _active_policy_keys(dt_mode: str):
    """Return policy keys that should be summarized for the requested dt mode."""
    keys = []
    for tag, _ in _requested_mode_pairs(dt_mode):
        keys.extend([f"ind_{tag}", f"glb_{tag}"])
    return keys


def _plot_per_patient(pid, results, save_dir, dt_mode="fixdt"):
    """Plot only the requested dt modes for one patient.

    This avoids empty variable-dt panels when the offline workflow is run with
    ``--dt_mode fixdt``.
    """
    os.makedirs(save_dir, exist_ok=True)

    for tag, title in _requested_mode_pairs(dt_mode):
        keys = [f"ind_{tag}", f"glb_{tag}"]
        keys = [key for key in keys if results.get(key) is not None]
        if not keys:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        for key in keys:
            mt = results[key]["sofa_mean"]
            st = results[key]["sofa_std"]
            c = COLORS.get(key, "#666666")
            ax.plot(T_GRID, mt, color=c, lw=1.8, label=LABELS.get(key, key))
            ax.fill_between(T_GRID, mt - st, mt + st, color=c, alpha=0.15)
        ax.axvline(POST_HOUR, color="gray", ls="--", lw=1.0)
        ax.set_xlabel("Time (h)", fontsize=13)
        ax.set_ylabel("SOFA score", fontsize=13)
        ax.set_title(f"SOFA — {title}", fontsize=13)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=11)

        ax = axes[1]
        for key in keys:
            mt = results[key]["lac_mean"]
            st = results[key]["lac_std"]
            c = COLORS.get(key, "#666666")
            ax.plot(T_GRID, mt, color=c, lw=1.8, label=LABELS.get(key, key))
            ax.fill_between(T_GRID, mt - st, mt + st, color=c, alpha=0.15)
        ax.axhline(LAC_DANGER_THR, color="red", ls="--", lw=1.2,
                   label=f"danger ({LAC_DANGER_THR} mmol/L)")
        ax.axhline(LAC_NORMAL_THR, color="orange", ls=":", lw=1.0,
                   label=f"normal ({LAC_NORMAL_THR} mmol/L)")
        ax.axvline(POST_HOUR, color="gray", ls="--", lw=1.0)
        ax.set_xlabel("Time (h)", fontsize=13)
        ax.set_ylabel("Lactate (mmol/L)", fontsize=13)
        ax.set_title(f"Lactate — {title}", fontsize=13)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=11)

        fig.suptitle(f"Patient {pid}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sofa_pid{pid}_{tag}.png"), dpi=150)
        plt.close(fig)


def _plot_summary(all_results: dict, save_dir: str, active_policies=None):
    """Summarize only policies that were actually requested/evaluated."""
    os.makedirs(save_dir, exist_ok=True)
    pids = sorted(all_results.keys())
    if not pids:
        raise RuntimeError("No collection results found. No valid patients/policies were evaluated.")

    if active_policies is None:
        active_policies = POLICIES
    active_policies = [
        key for key in active_policies
        if any(all_results[p].get(key) is not None for p in pids)
    ]
    if not active_policies:
        raise RuntimeError("No valid offline collection metrics found to summarize.")

    def _extr(key, metric):
        return np.array([
            all_results[p][key][metric]
            if all_results[p].get(key) is not None else np.nan
            for p in pids
        ], dtype=np.float32)

    metrics_by_policy = {}
    for key in active_policies:
        metrics_by_policy[key] = {
            "sofa": _extr(key, "mean_sofa"),
            "sofa_all": _extr(key, "mean_sofa_all"),
            "sr": _extr(key, "safety_rate"),
            "lac": _extr(key, "mean_lac"),
        }

    all_sofa = np.concatenate([
        vals["sofa"][~np.isnan(vals["sofa"])] for vals in metrics_by_policy.values()
    ])
    if all_sofa.size == 0:
        raise RuntimeError("No valid post-20h SOFA metrics found to summarize.")
    ymin_s = float(np.nanmin(all_sofa)) - 0.5
    ymax_s = float(np.nanmax(all_sofa)) + 0.5

    all_sofa_all = np.concatenate([
        vals["sofa_all"][~np.isnan(vals["sofa_all"])] for vals in metrics_by_policy.values()
    ])
    if all_sofa_all.size == 0:
        raise RuntimeError("No valid all-time SOFA metrics found to summarize.")
    ymin_sa = float(np.nanmin(all_sofa_all)) - 0.5
    ymax_sa = float(np.nanmax(all_sofa_all)) + 0.5

    xs = np.arange(len(pids))

    def _line_plot(fname, metric_name, ylabel, ymin=None, ymax=None, scale=1.0):
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in active_policies:
            vals = metrics_by_policy[key][metric_name] * scale
            ax.plot(xs, vals, color=COLORS.get(key, "#666666"), lw=1.5,
                    label=LABELS.get(key, key))
        ax.set_xlabel("Patient (sorted by ID)", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xticks([])
        ax.legend(fontsize=14)
        ax.tick_params(labelsize=13)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close(fig)

    _line_plot("mean_sofa_per_patient.png", "sofa",
               "Mean SOFA (t ≥ 20 h)", ymin_s, ymax_s)
    _line_plot("mean_sofa_alltime_per_patient.png", "sofa_all",
               "Mean SOFA (all time)", ymin_sa, ymax_sa)
    _line_plot("safety_rate.png", "sr",
               f"Safety rate (lac < {LAC_DANGER_THR:.0f}, all time) [%]",
               0, 105, scale=100.0)

    labels = [LABELS.get(k, k).replace(" ", "\n") for k in active_policies]
    colors = [COLORS.get(k, "#666666") for k in active_policies]

    def _strip(v):
        return v[~np.isnan(v)]

    def _boxplot(fname, metric_name, ylabel, ylim=None):
        data_list = [_strip(metrics_by_policy[k][metric_name]) for k in active_policies]
        if not any(len(v) > 0 for v in data_list):
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(data_list, patch_artist=True, tick_labels=labels)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close(fig)

    _boxplot("boxplot_sofa_post20h.png", "sofa", "Mean SOFA (t ≥ 20 h)", (ymin_s, ymax_s))
    _boxplot("boxplot_sofa_alltime.png", "sofa_all", "Mean SOFA (all time)", (ymin_sa, ymax_sa))
    _boxplot("boxplot_safety_rate.png", "sr", f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]", (0, 1.05))

    stat_keys = [
        (key,
         metrics_by_policy[key]["sofa"],
         metrics_by_policy[key]["sofa_all"],
         metrics_by_policy[key]["sr"])
        for key in active_policies
    ]
    sofa_means = [np.nanmean(v) for _, v, _, _ in stat_keys]
    sofa_stds = [np.nanstd(v) for _, v, _, _ in stat_keys]
    sofa_all_means = [np.nanmean(va) for _, _, va, _ in stat_keys]
    sofa_all_stds = [np.nanstd(va) for _, _, va, _ in stat_keys]
    sr_means = [np.nanmean(r) * 100 for _, _, _, r in stat_keys]
    sr_stds = [np.nanstd(r) * 100 for _, _, _, r in stat_keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xs_bar = np.arange(len(labels))

    ax = axes[0]
    ax.bar(xs_bar, sofa_means, yerr=sofa_stds, color=colors, alpha=0.8,
           capsize=5, error_kw={"lw": 1.5})
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean SOFA (t ≥ 20 h)", fontsize=14)
    ax.tick_params(labelsize=12)

    ax = axes[1]
    ax.bar(xs_bar, sofa_all_means, yerr=sofa_all_stds, color=colors, alpha=0.8,
           capsize=5, error_kw={"lw": 1.5})
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean SOFA (all time)", fontsize=14)
    ax.tick_params(labelsize=12)

    ax = axes[2]
    ax.bar(xs_bar, sr_means, yerr=sr_stds, color=colors, alpha=0.8,
           capsize=5, error_kw={"lw": 1.5})
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]", fontsize=14)
    ax.set_ylim(0, 115)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overall_stats.png"), dpi=150)
    plt.close(fig)

    lines = ["policy,n,sofa_post20h_mean,sofa_post20h_std,sofa_alltime_mean,sofa_alltime_std,safety_rate_mean,safety_rate_std"]
    print("\n" + "=" * 80)
    print(f"{'Policy':<20} {'N':>4} {'SOFA(≥20h)':>11} {'±':>3} {'std':>6}  "
          f"{'SOFA(all)':>10} {'±':>3} {'std':>6}  "
          f"{'SafeRate%':>10} {'±':>3} {'std':>6}")
    print("-" * 80)
    for key, sofa_v, sofa_va, sr_v in stat_keys:
        n = int(np.sum(~np.isnan(sofa_v)))
        sm = np.nanmean(sofa_v)
        ss = np.nanstd(sofa_v)
        sam = np.nanmean(sofa_va)
        sas = np.nanstd(sofa_va)
        rm = np.nanmean(sr_v) * 100
        rs = np.nanstd(sr_v) * 100
        print(f"{key:<20} {n:>4} {sm:>11.3f} {'±':>3} {ss:>6.3f}  "
              f"{sam:>10.3f} {'±':>3} {sas:>6.3f}  {rm:>10.1f} {'±':>3} {rs:>6.1f}")
        lines.append(f"{key},{n},{sm:.4f},{ss:.4f},{sam:.4f},{sas:.4f},{rm:.2f},{rs:.2f}")
    print("=" * 80 + "\n")

    with open(os.path.join(save_dir, "overall_stats.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    summary = {"pids": pids}
    for key, vals in metrics_by_policy.items():
        summary[f"{key}_sofa"] = vals["sofa"]
        summary[f"{key}_sofa_all"] = vals["sofa_all"]
        summary[f"{key}_sr"] = vals["sr"]
        summary[f"{key}_lac"] = vals["lac"]

    np.save(os.path.join(save_dir, "summary_scores.npy"), summary)


# ─── Global dataset aggregation ───────────────────────────────────────────────

def _merge_and_save_global_datasets(glb_datasets_per_patient: dict,
                                    dataset_dir: str,
                                    action_noise_std: float,
                                    dt_mode: str = "fixdt"):

    """Concatenate per-patient global-policy datasets and save as trpo_pop_*.npz.

New function.

    glb_datasets_per_patient:
        { pid: { "fixdt": dataset_dict, "vardt": dataset_dict }, ... }

    The merged dataset keeps the same keys as per-patient ones.
    ep_id is re-assigned globally (0 … total_eps-1) so downstream CQL code
    can iterate over episodes unambiguously.
    """
    os.makedirs(dataset_dir, exist_ok=True)

    for tag, _ in _requested_mode_pairs(dt_mode):
        fragments = []
        for pid, ds_map in sorted(glb_datasets_per_patient.items()):
            if tag in ds_map and ds_map[tag] is not None:
                fragments.append((pid, ds_map[tag]))

        if not fragments:
            print(f"[trpo_pop] no data for {tag}, skipping.", flush=True)
            continue

        # --- concatenate all fields across patients -----------------------
        def _cat_field(key):
            return np.concatenate([d[key] for _, d in fragments], axis=0)

        # Re-index ep_id continuously across patients
        all_ep_ids = []
        ep_offset  = 0
        for _, d in fragments:
            local_ids  = d["meta/ep_id"]                        # 0-based per patient
            n_eps_here = int(d["meta/ep_length"].shape[0])
            all_ep_ids.append(local_ids + ep_offset)
            ep_offset += n_eps_here

        merged = {
            "transitions/s":      _cat_field("transitions/s"),
            "transitions/a":      _cat_field("transitions/a"),
            "transitions/r":      _cat_field("transitions/r"),
            "transitions/ns":     _cat_field("transitions/ns"),
            "transitions/done":   _cat_field("transitions/done"),
            "meta/sofa":          _cat_field("meta/sofa"),
            "meta/lactate":       _cat_field("meta/lactate"),
            "meta/ep_id":         np.concatenate(all_ep_ids, axis=0),
            "meta/ep_length":     _cat_field("meta/ep_length"),
            "meta/ep_reward_sum": _cat_field("meta/ep_reward_sum"),
            # record which patient each episode came from
            "meta/source_pids":   np.array(
                [pid for pid, d in fragments
                 for _ in range(d["meta/ep_length"].shape[0])],
                dtype=np.int32,
            ),
            "info/n_patients":       np.array([len(fragments)]),
            "info/n_transitions":    np.array([len(_cat_field("transitions/r"))]),
            "info/action_noise_std": np.array([action_noise_std]),
            "info/state_dim":        fragments[0][1]["info/state_dim"],
            "info/action_dim":       fragments[0][1]["info/action_dim"],
            "info/policy_tag":       np.array([f"global_lagrangian_trpo_{tag}"]),
        }

        out_path = os.path.join(dataset_dir, f"trpo_pop_{tag}_dataset.npz")
        np.savez_compressed(out_path, **merged)
        n = int(merged["info/n_transitions"][0])
        print(f"  [NPZ saved] {out_path}  ({n:,} transitions,  "
              f"{len(fragments)} patients)", flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Compare per-patient Lagrangian TRPO vs global on individual patient PINNs"
    )
    p.add_argument("--csv",       default="data/mimic_pinn_v4_filtered.csv",
                   help="MIMIC CSV for patient list + init fallback (omit with --no_mimic_csv).")
    p.add_argument(
        "--no_mimic_csv",
        action="store_true",
        help="Discover patients from --pinn_dir only; require init_state_norm.npy per patient.",
    )
    p.add_argument("--pinn_dir",  default="results/pinn/individual")
    p.add_argument("--glb_scales", default="results/pinn/population/scales.npy")
    p.add_argument("--ind_root",  default="results/online/individual")
    p.add_argument("--glb_root",  default="results/online/population")
    p.add_argument("--save_dir",  default="results/offline")
    p.add_argument("--behavior_algo", type=str, default="lagrangian_trpo")
    p.add_argument("--dt_mode", type=str, default="fixdt",
               choices=["fixdt", "vardt", "both"])
    p.add_argument("--n_eval",    type=int, default=5)
    p.add_argument("--K",         type=int, default=20)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--ind_hidden", type=int, default=256)
    p.add_argument("--glb_hidden", type=int, default=256)
    p.add_argument("--patient_ids", type=int, nargs="*", default=None)
                #    default=[200325, 201046, 201101, 202028, 202515])  # todo：all patient    default=None
    p.add_argument("--patient_ids_file", type=str, default=None, help="Optional pid-only CSV/TXT file, e.g. checkpoints-cohort/cohort_1/cohort_1_training.csv. "
         "If provided, it is used instead of --patient_ids for individual patient selection.",)

    p.add_argument("--action_noise_std", type=float, default=0.05,
                   help="Std of Gaussian noise added to each action before env.step(). "
                        "Clipped to [-1, 1] afterwards.  0 = no noise (original behaviour). "
                        "Recommended range: 0.02–0.10.")
    return p.parse_args()

def resolve_scales_path(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    for name in ["scales.npy", "cluster_scales.npy"]:
        cand = p / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No scales.npy or cluster_scales.npy found under {p}")

def main():
    args = _parse()

    sys.path.insert(0, str(_HERE.parent))

    from pinn_bundle import discover_patient_ids_from_pinn_dir, load_patient_ids_from_file

    pinn_base = Path(args.pinn_dir)
    requested_pids = None
    if args.patient_ids_file:
        requested_pids = load_patient_ids_from_file(args.patient_ids_file)
    elif args.patient_ids:
        requested_pids = list(args.patient_ids)

    if args.no_mimic_csv:
        discovered = discover_patient_ids_from_pinn_dir(pinn_base)
        if requested_pids is not None:
            available = set(discovered)
            missing = [p for p in requested_pids if p not in available]
            if missing:
                print(
                    f"[warning] {len(missing)} requested patients do not have PINNs under "
                    f"{args.pinn_dir}; first missing IDs: {missing[:10]}",
                    flush=True,
                )
            pids = [p for p in requested_pids if p in available]
        else:
            pids = discovered
    else:
        bundle = load_all_patients(args.csv)
        all_pats = sorted(
            [p for p in bundle.patients
            if (pinn_base / f"patient_{p['icu_id']}" / "pinn.pt").exists()],
            key=lambda p: p["icu_id"],
        )
        if requested_pids is not None:
            want = set(requested_pids)
            all_pats = [p for p in all_pats if p["icu_id"] in want]
            available = {p["icu_id"] for p in all_pats}
            pids = [p for p in requested_pids if p in available]
        else:
            pids = [p["icu_id"] for p in all_pats]

    if not pids:
        raise RuntimeError(
            f"No patient PINNs found under {args.pinn_dir} for the requested patient_ids."
        )

    per_patient_dir = os.path.join(args.save_dir, "per_patient")
    summary_dir     = os.path.join(args.save_dir, "summary")
    dataset_dir     = os.path.join(args.save_dir, "datasets")
    os.makedirs(per_patient_dir, exist_ok=True)
    os.makedirs(summary_dir,     exist_ok=True)
    os.makedirs(dataset_dir,     exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  eval_ind_lagrangian_trpo")
    print(f"  pinn_dir          : {args.pinn_dir}")
    print(f"  ind_root          : {args.ind_root}")
    print(f"  glb_root          : {args.glb_root}")
    print(f"  save_dir          : {args.save_dir}")
    print(f"  dataset_dir       : {dataset_dir}")
    print(f"  patient_ids_file  : {args.patient_ids_file}")
    print(f"  n_eval            : {args.n_eval}  K={args.K}")
    print(f"  ind_hidden        : {args.ind_hidden}  glb_hidden={args.glb_hidden}")
    print(f"  action_noise_std  : {args.action_noise_std}")
    print(f"  patients          : {len(pids)}")
    print(f"  no_mimic_csv      : {args.no_mimic_csv}")
    print(f"{'='*60}\n")

    csv_arg = None if args.no_mimic_csv else args.csv
    all_results = {}
    glb_datasets_per_patient = {}

    for pid in pids:

        glb_scales_path = resolve_scales_path(args.glb_scales)
        ret = _worker(
            pid, args.pinn_dir, glb_scales_path, args.ind_root, args.glb_root,
            per_patient_dir, args.n_eval, args.K, args.device, csv_arg,
            ind_hidden=args.ind_hidden, glb_hidden=args.glb_hidden,
            action_noise_std=args.action_noise_std,
            dataset_dir=dataset_dir,
            behavior_algo=args.behavior_algo,
            dt_mode=args.dt_mode,
        )
        if ret is None:
            continue

        results, glb_ds = ret
        all_results[pid] = results
        glb_datasets_per_patient[pid] = glb_ds
        _plot_per_patient(pid, results, per_patient_dir, dt_mode=args.dt_mode)

    if not all_results:
        raise RuntimeError(
            "No valid offline collection results were produced. Check PINN and policy paths."
        )

    _plot_summary(all_results, summary_dir, active_policies=_active_policy_keys(args.dt_mode))


    _merge_and_save_global_datasets(
        glb_datasets_per_patient,
        dataset_dir,
        action_noise_std=args.action_noise_std,
        dt_mode=args.dt_mode,
    )

    print(f"\nDone.  Plots  → {args.save_dir}")
    print(f"       Datasets → {dataset_dir}")


if __name__ == "__main__":
    main()
