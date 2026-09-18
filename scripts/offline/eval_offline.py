"""eval_offline.py — Evaluate trained offline RL models on patient PINNs.

This script evaluates trained offline policies on individual patient PINN
simulators. It supports two evaluation scopes:
  - individual: per-patient offline policies, e.g. pid_<pid>_fixdt
  - population_transfer: one population-trained offline policy, e.g. trpo_pop_fixdt,
    evaluated on each individual patient simulator

No-MIMIC workflow:
  Use --no_mimic_csv with released patient PINN folders. Initial states are
  loaded from each patient's `init_state_norm.npy`; the MIMIC-derived CSV is
  only used as a development fallback when that file is unavailable.

Patient selection priority:
  1. --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv
  2. --patient_ids 200325 201046 ...
  3. all discoverable patient_<pid>/ folders under --pinn_dir

Saved rollout datasets:
  <save_dir>/datasets/pid_<pid>_<method>_fixdt_dataset.npz
  <save_dir>/datasets/<method>_pop_fixdt_dataset.npz

Usage:
    python3 scripts/offline/eval_offline.py \
        --no_mimic_csv \
        --eval_scope individual \
        --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv \
        --pinn_dir results/pinn/individual \
        --cql_model_dir results/offline/policies/individual/cql/models \
        --glb_scales results/pinn/population/scales.npy \
        --save_dir results/offline/eval/individual/cql \
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

POST_HOUR      = 20.0
TOTAL_TIME_H   = 96.0
T_GRID         = np.arange(0, TOTAL_TIME_H + 1, 1.0)
LAC_DANGER_THR = 4.0
LAC_NORMAL_THR = 2.0

STATE_DIM  = 6
ACTION_DIM = 3

POLICIES = ["ind_fixdt", "glb_fixdt"]
LABELS = {
    "ind_fixdt": "ind CQL (fixed-dt)",
    "glb_fixdt": "global CQL (fixed-dt)",
}
COLORS = {
    "ind_fixdt": "#1f77b4",
    "glb_fixdt": "#d62728",
}


# ─── CQL Actor (must match train_cql_multi.py / train_ddpg_cql.py) ────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.fc1    = nn.Linear(state_dim, hidden)
        self.bn1    = nn.BatchNorm1d(hidden)
        self.fc2    = nn.Linear(hidden, hidden)
        self.bn2    = nn.BatchNorm1d(hidden)
        self.fc_out = nn.Linear(hidden, action_dim)

    def forward(self, s):
        x = F.leaky_relu(self.bn1(self.fc1(s)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        return torch.tanh(self.fc_out(x))

    def select_action(self, state_np: np.ndarray) -> np.ndarray:
        """Deterministic inference from a numpy state vector."""
        self.eval()
        with torch.no_grad():
            s = torch.tensor(state_np[:STATE_DIM], dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
            a = self(s).squeeze(0).cpu().numpy()
        self.train()
        return a


# ─── AdaptedEnv ───────────────────────────────────────────────────────────────

class AdaptedEnv:
    """Wrap a per-patient ICUEnvironment for evaluation with a globally-trained CQL.

    Mirrors the AdaptedEnv in eval_ind_lagrangian_trpo.py:
    converts per-patient-normalised physical states into the global
    normalisation space so that the population model sees the same
    distribution it was trained on.
    """

    def __init__(self, patient_env, mean_glob: np.ndarray, std_glob: np.ndarray,
                 mean_pat: np.ndarray, std_pat: np.ndarray):
        self._env      = patient_env
        self.mean_glob = mean_glob
        self.std_glob  = std_glob
        self.mean_pat  = mean_pat
        self.std_pat   = std_pat

    def _adapt_state(self, state: np.ndarray) -> np.ndarray:
        """Re-normalise physical dims; pass extra dims (t, k …) through as-is."""
        phys_pat  = state[:STATE_DIM]
        phys_real = phys_pat * self.std_pat[:STATE_DIM] + self.mean_pat[:STATE_DIM]
        phys_glob = (phys_real - self.mean_glob[:STATE_DIM]) / self.std_glob[:STATE_DIM]
        return np.concatenate([phys_glob, state[STATE_DIM:]], dtype=np.float32)

    def reset(self, **kwargs):
        return self._adapt_state(self._env.reset(**kwargs))

    def step(self, action):
        next_state, reward, done, info = self._env.step(action)
        return self._adapt_state(next_state), reward, done, info

    @property
    def current_t(self):
        return self._env.current_t


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _interp_zoh(t_arr, val_arr, t_grid):
    out = np.empty(len(t_grid), dtype=np.float32)
    for gi, tg in enumerate(t_grid):
        idx = max(0, int(np.searchsorted(t_arr, tg, side="right")) - 1)
        out[gi] = val_arr[min(idx, len(val_arr) - 1)]
    return out


def _load_actor(pt_path: str, state_dim: int = STATE_DIM,
                action_dim: int = ACTION_DIM, hidden: int = 256) -> Actor:
    actor = Actor(state_dim, action_dim, hidden).to(DEVICE)
    sd = torch.load(pt_path, map_location=DEVICE)
    # support dict with 'actor' key or raw state-dict
    if isinstance(sd, dict) and "fc1.weight" not in sd:
        sd = sd.get("actor", sd)
    actor.load_state_dict(sd)
    actor.eval()
    return actor


# ─── Episode rollout ──────────────────────────────────────────────────────────

def _run_episode(actor: Actor, env, init_state_norm,
                 action_noise_std: float = 0.0,
                 rng: np.random.Generator = None):
    """Run one episode and return interpolated trajectories, metrics, transitions."""
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(init_state_norm, dtype=np.float32)
    state = env.reset(init_state_norm=x0)
    done  = False

    t_list, sofa_list, lac_list = [], [], []
    trans_s, trans_a, trans_r = [], [], []
    trans_ns, trans_done      = [], []
    trans_sofa, trans_lac     = [], []

    while not done:
        t0     = float(env.current_t)
        action = actor.select_action(state)

        if action_noise_std > 0.0:
            noise  = rng.normal(0.0, action_noise_std, size=action.shape).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)

        next_state, reward, done, info = env.step(action)

        t_list.append(t0)
        sofa_list.append(float(info["sofa"]))
        lac_list.append(float(info["lactate"]))

        trans_s.append(state[:STATE_DIM].copy())
        trans_a.append(action[:ACTION_DIM].copy())
        trans_r.append(float(reward))
        trans_ns.append(next_state[:STATE_DIM].copy())
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


def _collect_policy(actor, env, init_state_norm, n_eval,
                    action_noise_std=0.0, rng=None):
    """Run n_eval episodes; return aggregated stats + offline dataset dict."""
    if rng is None:
        rng = np.random.default_rng()

    sofa_trajs, lac_trajs = [], []
    all_metrics = {k: [] for k in ["mean_sofa", "mean_sofa_all", "mean_lac",
                                    "safety_rate", "clearance_6h", "time_above_4"]}
    all_s, all_a, all_r, all_ns, all_done = [], [], [], [], []
    all_sofa_steps, all_lac_steps, all_ep_id = [], [], []
    ep_lengths, ep_reward_sum = [], []

    for ep_idx in range(n_eval):
        sofa_t, lac_t, m, trans = _run_episode(
            actor, env, init_state_norm,
            action_noise_std=action_noise_std, rng=rng)

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
        "transitions/s":         _cat(all_s),
        "transitions/a":         _cat(all_a),
        "transitions/r":         _cat(all_r),
        "transitions/ns":        _cat(all_ns),
        "transitions/done":      _cat(all_done),
        "meta/sofa":             _cat(all_sofa_steps),
        "meta/lactate":          _cat(all_lac_steps),
        "meta/ep_id":            _cat(all_ep_id),
        "meta/ep_length":        np.array(ep_lengths,    dtype=np.int32),
        "meta/ep_reward_sum":    np.array(ep_reward_sum, dtype=np.float32),
        "info/n_eval":           np.array([n_eval]),
        "info/action_noise_std": np.array([action_noise_std]),
        "info/state_dim":        np.array([STATE_DIM]),
        "info/action_dim":       np.array([ACTION_DIM]),
        "info/n_transitions":    np.array([int(_cat(all_r).shape[0])]),
    }

    return (sofa_mat.mean(0), sofa_mat.std(0),
            lac_mat.mean(0),  lac_mat.std(0),
            {k: float(np.mean(v)) for k, v in all_metrics.items()},
            dataset)


# ─── Dataset saving ───────────────────────────────────────────────────────────

def _save_dataset(path, dataset, pid=None, policy_tag="", action_noise_std=0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = dict(dataset)
    if pid is not None:
        ds["info/patient_id"] = np.array([pid])
    ds["info/policy_tag"]  = np.array([policy_tag])
    np.savez_compressed(path, **ds)
    n = int(ds["info/n_transitions"][0])
    print(f"  [NPZ saved] {path}  ({n:,} transitions)", flush=True)


# ─── Per-patient worker ───────────────────────────────────────────────────────

def _infer_method_name(save_dir=None, model_dir=None, explicit=None):
    """Infer a short method slug used in evaluation dataset filenames."""
    if explicit:
        return str(explicit).lower()
    known = {"dqn", "cql", "gcql"}
    for raw in (save_dir, model_dir):
        if raw is None:
            continue
        parts = [p.lower() for p in Path(raw).parts]
        for part in reversed(parts):
            if part in known:
                return part
    return "offline"


def _worker(pid, pinn_dir, cql_model_dir, save_dir, csv_path,
            n_eval, K, hidden, action_noise_std, dataset_dir, dt_modes=("fixdt",),
            glb_scales_path=None, model_tag=None, eval_scope="both", method_name="offline"):

    from models      import PINN
    from rl          import ICUEnvironment, SMDP_STATE_DIM, OPTION_ACTION_DIM
    from data.config import state_dim as sd, action_dim as ad, device as global_device

    from pinn_bundle import load_init_and_scales_for_patient

    rng = np.random.default_rng()

    # Load global population normalization scales for population-transfer evaluation.
    mean_glob = std_glob = None
    if glb_scales_path is not None and Path(glb_scales_path).exists():
        sc_glob   = np.load(glb_scales_path, allow_pickle=True).item()
        mean_glob = sc_glob["mean"].astype(np.float32)
        std_glob  = sc_glob["std"].astype(np.float32)
    elif glb_scales_path is not None:
        print(f"[patient {pid}] WARNING: --glb_scales path not found "
              f"({glb_scales_path}). Population-transfer policies will use "
              "per-patient normalisation.",
              flush=True)

    try:
        init_norm, pinn_path, mean_pat, std_pat, asc_pat, state_min_np, state_max_np = (
            load_init_and_scales_for_patient(Path(pinn_dir), pid, csv_path)
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[patient {pid}] {e}, skipping.", flush=True)
        return None

    from models import PINN as PINNModel
    pinn = PINNModel(sd, ad).to(global_device)
    pinn.load_state_dict(torch.load(str(pinn_path), map_location=global_device))
    pinn.eval()

    fixed_dt = TOTAL_TIME_H / K

    def make_env(use_dt):
        return ICUEnvironment(
            pinn_model      = pinn,
            mean_np         = mean_pat,
            std_np          = std_pat,
            action_min_norm = np.zeros(ad, dtype=np.float32),
            action_max_norm = np.ones(ad,  dtype=np.float32),
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

    def make_glb_env(use_dt):
        """Patient simulator wrapped with population normalization for shared models."""
        pat_env = make_env(use_dt)
        if mean_glob is not None and std_glob is not None:
            return AdaptedEnv(pat_env, mean_glob, std_glob, mean_pat, std_pat)
        return pat_env

    results = {}
    glb_datasets = {}

    def _run_and_store(key, actor, env):
        sofa_mean, sofa_std, lac_mean, lac_std, m, ds = _collect_policy(
            actor, env, init_norm, n_eval,
            action_noise_std=action_noise_std, rng=rng)
        results[key] = {
            "sofa_mean": sofa_mean, "sofa_std": sofa_std,
            "lac_mean":  lac_mean,  "lac_std":  lac_std,
            "_dataset":  ds,
            **m,
        }
        print(
            f"[patient {pid}] {key}  "
            f"sofa={m['mean_sofa']:.2f}  lac={m['mean_lac']:.2f}  "
            f"safety={m['safety_rate']:.2%}  clr6h={m['clearance_6h']:.1%}  "
            f"t>4h={m['time_above_4']:.1f}h",
            flush=True,
        )

    mode_pairs = []
    if "fixdt" in dt_modes:
        mode_pairs.append(("fixdt", False))
    if "vardt" in dt_modes:
        mode_pairs.append(("vardt", True))

    # Individual offline policies: one per-patient model per patient.
    if eval_scope in ("individual", "both"):
        for tag, use_dt in mode_pairs:
            key = f"ind_{tag}"
            subdir = f"pid_{pid}_{tag}"
            pt_path = Path(cql_model_dir) / subdir / "actor_final.pt"
            if not pt_path.exists():
                print(f"[patient {pid}] {key}: model not found ({pt_path})", flush=True)
                results[key] = None
                continue

            actor = _load_actor(str(pt_path), hidden=hidden)
            _run_and_store(key, actor, make_env(use_dt))

            if results[key] is not None and dataset_dir is not None:
                _save_dataset(
                    os.path.join(dataset_dir, f"pid_{pid}_{method_name}_{tag}_dataset.npz"),
                    results[key]["_dataset"],
                    pid=pid, policy_tag=f"ind_{method_name}_{tag}",
                    action_noise_std=action_noise_std,
                )

    # Population-transfer offline policy: one shared model evaluated on each patient PINN.
    if eval_scope in ("population_transfer", "both"):
        for tag, use_dt in mode_pairs:
            key = f"glb_{tag}"
            subdir = model_tag if model_tag is not None else f"trpo_pop_{tag}"
            pt_path = Path(cql_model_dir) / subdir / "actor_final.pt"
            if not pt_path.exists():
                print(f"[patient {pid}] {key}: model not found ({pt_path})", flush=True)
                results[key] = None
                continue

            actor = _load_actor(str(pt_path), hidden=hidden)
            _run_and_store(key, actor, make_glb_env(use_dt))

            if results[key] is not None:
                glb_datasets[tag] = results[key]["_dataset"]

    return results, glb_datasets


# ─── Aggregate global datasets ────────────────────────────────────────────────

def _merge_and_save_global(
    glb_datasets_per_patient,
    dataset_dir,
    action_noise_std,
    method_name="offline",
    dt_modes=("fixdt",),
):
    """Aggregate population-transfer rollout datasets across patients."""
    os.makedirs(dataset_dir, exist_ok=True)

    for tag in dt_modes:
        fragments = [
            (pid, ds[tag])
            for pid, ds in sorted(glb_datasets_per_patient.items())
            if tag in ds and ds[tag] is not None
        ]
        if not fragments:
            print(f"[{method_name}_pop] no data for {tag}, skipping.", flush=True)
            continue

        def _cat(key):
            return np.concatenate([d[key] for _, d in fragments], axis=0)

        ep_ids = []
        offset = 0
        for _, d in fragments:
            n_eps = int(d["meta/ep_length"].shape[0])
            ep_ids.append(d["meta/ep_id"] + offset)
            offset += n_eps

        merged = {
            "transitions/s":         _cat("transitions/s"),
            "transitions/a":         _cat("transitions/a"),
            "transitions/r":         _cat("transitions/r"),
            "transitions/ns":        _cat("transitions/ns"),
            "transitions/done":      _cat("transitions/done"),
            "meta/sofa":             _cat("meta/sofa"),
            "meta/lactate":          _cat("meta/lactate"),
            "meta/ep_id":            np.concatenate(ep_ids, axis=0),
            "meta/ep_length":        _cat("meta/ep_length"),
            "meta/ep_reward_sum":    _cat("meta/ep_reward_sum"),
            "meta/source_pids":      np.array(
                [pid for pid, d in fragments
                 for _ in range(d["meta/ep_length"].shape[0])], dtype=np.int32),
            "info/n_patients":       np.array([len(fragments)]),
            "info/n_transitions":    np.array([int(_cat("transitions/r").shape[0])]),
            "info/action_noise_std": np.array([action_noise_std]),
            "info/state_dim":        np.array([STATE_DIM]),
            "info/action_dim":       np.array([ACTION_DIM]),
            "info/policy_tag":       np.array([f"global_{method_name}_{tag}"]),
        }

        out_path = os.path.join(dataset_dir, f"{method_name}_pop_{tag}_dataset.npz")
        np.savez_compressed(out_path, **merged)
        n = int(merged["info/n_transitions"][0])
        print(f"  [NPZ saved] {out_path}  ({n:,} transitions, "
              f"{len(fragments)} patients)", flush=True)


# ─── Per-patient plots ────────────────────────────────────────────────────────

def _plot_per_patient(pid, results, save_dir, active_policies=None):
    os.makedirs(save_dir, exist_ok=True)

    if active_policies is None:
        active_policies = POLICIES

    keys = [k for k in active_policies if results.get(k) is not None]
    if not keys:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    for key in keys:
        mt, st = results[key]["sofa_mean"], results[key]["sofa_std"]
        c = COLORS.get(key, "#666666")
        ax.plot(T_GRID, mt, color=c, lw=1.8, label=LABELS.get(key, key))
        ax.fill_between(T_GRID, mt - st, mt + st, color=c, alpha=0.15)
    ax.axvline(POST_HOUR, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Time (h)", fontsize=13)
    ax.set_ylabel("SOFA score", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)

    ax = axes[1]
    for key in keys:
        mt, st = results[key]["lac_mean"], results[key]["lac_std"]
        c = COLORS.get(key, "#666666")
        ax.plot(T_GRID, mt, color=c, lw=1.8, label=LABELS.get(key, key))
        ax.fill_between(T_GRID, mt - st, mt + st, color=c, alpha=0.15)
    ax.axhline(LAC_DANGER_THR, color="red",    ls="--", lw=1.2,
               label=f"danger ({LAC_DANGER_THR} mmol/L)")
    ax.axhline(LAC_NORMAL_THR, color="orange", ls=":",  lw=1.0,
               label=f"normal ({LAC_NORMAL_THR} mmol/L)")
    ax.axvline(POST_HOUR, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Time (h)", fontsize=13)
    ax.set_ylabel("Lactate (mmol/L)", fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cql_sofa_pid{pid}_fixdt.png"), dpi=150)
    plt.close(fig)


# ─── Summary plots ────────────────────────────────────────────────────────────

def _plot_summary(all_results, save_dir, active_policies=None):
    os.makedirs(save_dir, exist_ok=True)
    pids = sorted(all_results.keys())
    if not pids:
        raise RuntimeError("No evaluation results found. No valid patients/models were evaluated.")

    if active_policies is None:
        active_policies = POLICIES
    active_policies = [
        k for k in active_policies
        if any(all_results[p].get(k) is not None for p in pids)
    ]
    if not active_policies:
        raise RuntimeError("No valid fixed-dt evaluation metrics found to summarize.")

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
        v["sofa"][~np.isnan(v["sofa"])] for v in metrics_by_policy.values()
    ])
    if all_sofa.size == 0:
        raise RuntimeError("No valid post-20h SOFA metrics found to summarize.")
    ymin_s = float(np.nanmin(all_sofa)) - 0.5
    ymax_s = float(np.nanmax(all_sofa)) + 0.5

    all_sofa_all = np.concatenate([
        v["sofa_all"][~np.isnan(v["sofa_all"])] for v in metrics_by_policy.values()
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
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=13)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close(fig)

    _line_plot("cql_mean_sofa_fixdt.png", "sofa",
               "Mean SOFA (t ≥ 20 h)", ymin_s, ymax_s)
    _line_plot("cql_mean_sofa_alltime_fixdt.png", "sofa_all",
               "Mean SOFA (all time)", ymin_sa, ymax_sa)
    _line_plot("cql_safety_rate_fixdt.png", "sr",
               f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]",
               0, 105, scale=100.0)

    labels = [LABELS.get(k, k).replace(" ", "\\n") for k in active_policies]
    colors = [COLORS.get(k, "#666666") for k in active_policies]

    def _strip(v): return v[~np.isnan(v)]

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

    _boxplot("cql_boxplot_sofa_post20h.png", "sofa",
             "Mean SOFA (t ≥ 20 h)", (ymin_s, ymax_s))
    _boxplot("cql_boxplot_sofa_alltime.png", "sofa_all",
             "Mean SOFA (all time)", (ymin_sa, ymax_sa))
    _boxplot("cql_boxplot_safety_rate.png", "sr",
             f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]", (0, 1.05))

    stat_keys = [
        (key,
         metrics_by_policy[key]["sofa"],
         metrics_by_policy[key]["sofa_all"],
         metrics_by_policy[key]["sr"])
        for key in active_policies
    ]
    bar_labels = [LABELS.get(k, k).replace(" ", "\\n") for k in active_policies]
    sofa_means = [np.nanmean(v) for _, v, _, _ in stat_keys]
    sofa_stds = [np.nanstd(v) for _, v, _, _ in stat_keys]
    sofa_all_means = [np.nanmean(va) for _, _, va, _ in stat_keys]
    sofa_all_stds = [np.nanstd(va) for _, _, va, _ in stat_keys]
    sr_means = [np.nanmean(r) * 100 for _, _, _, r in stat_keys]
    sr_stds = [np.nanstd(r) * 100 for _, _, _, r in stat_keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xs_bar = np.arange(len(bar_labels))

    ax = axes[0]
    ax.bar(xs_bar, sofa_means, yerr=sofa_stds, color=colors, alpha=0.8,
           capsize=5, error_kw={"lw": 1.5})
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel("Mean SOFA (t ≥ 20 h)", fontsize=14)
    ax.tick_params(labelsize=12)

    ax = axes[1]
    ax.bar(xs_bar, sofa_all_means, yerr=sofa_all_stds, color=colors, alpha=0.8,
           capsize=5, error_kw={"lw": 1.5})
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel("Mean SOFA (all time)", fontsize=14)
    ax.tick_params(labelsize=12)

    ax = axes[2]
    ax.bar(xs_bar, sr_means, yerr=sr_stds, color=colors, alpha=0.8,
           capsize=5, error_kw={"lw": 1.5})
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel(f"Safety rate (lac < {LAC_DANGER_THR:.0f}) [%]", fontsize=14)
    ax.set_ylim(0, 115)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cql_overall_stats.png"), dpi=150)
    plt.close(fig)

    # Print and save CSV.
    lines = ["policy,n,sofa_post20h_mean,sofa_post20h_std,sofa_alltime_mean,"
             "sofa_alltime_std,safety_rate_mean,safety_rate_std"]
    print("\n" + "=" * 80)
    print(f"{'Policy':<20} {'N':>4} {'SOFA(≥20h)':>11} {'±':>3} {'std':>6}  "
          f"{'SOFA(all)':>10} {'±':>3} {'std':>6}  "
          f"{'SafeRate%':>10} {'±':>3} {'std':>6}")
    print("-" * 80)
    for (key, sofa_v, sofa_va, sr_v) in stat_keys:
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

    with open(os.path.join(save_dir, "cql_overall_stats.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    summary = {"pids": pids}
    for key, vals in metrics_by_policy.items():
        summary[f"{key}_sofa"] = vals["sofa"]
        summary[f"{key}_sofa_all"] = vals["sofa_all"]
        summary[f"{key}_sr"] = vals["sr"]
        summary[f"{key}_lac"] = vals["lac"]

    # Backward-compatible aliases used by the current plotting script when
    # reading a population-transfer summary directory separately.
    if active_policies == ["glb_fixdt"] and "glb_fixdt" in metrics_by_policy:
        vals = metrics_by_policy["glb_fixdt"]
        summary["ind_fixdt_sofa"] = vals["sofa"]
        summary["ind_fixdt_sofa_all"] = vals["sofa_all"]
        summary["ind_fixdt_sr"] = vals["sr"]
        summary["ind_fixdt_lac"] = vals["lac"]

    np.save(os.path.join(save_dir, "cql_summary_scores.npy"), summary)
    np.save(os.path.join(save_dir, "summary_scores.npy"), summary)


# ─── argparse ────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Evaluate trained CQL models on individual patient PINNs"
    )
    p.add_argument("--csv",           default="data/mimic_pinn_v4_filtered.csv",
                   help="MIMIC CSV for patient list + init fallback (omit with --no_mimic_csv).")
    p.add_argument(
        "--no_mimic_csv",
        action="store_true",
        help="Discover patients from --pinn_dir only; require init_state_norm.npy per patient.",
    )
    p.add_argument("--pinn_dir",      default="results/pinn/individual")
    p.add_argument("--cql_model_dir", default="results/offline/policies/individual/cql/models",
                   help="Root directory of trained CQL models (contains pid_*/  and trpo_pop_*/ sub-dirs)")

    p.add_argument("--glb_scales",
                   default="results/pinn/population/scales.npy",
                   help="Path to cluster_scales.npy used for population CQL training. "
                        "Provides mean/std to renormalise patient states for global model. "
                        "If not supplied (or path missing), global policies fall back to "
                        "per-patient normalisation.")
    p.add_argument("--save_dir",      default="results/offline/eval/individual/cql")
    p.add_argument("--model_tag",     type=str, default=None,
                   help="If set, evaluate the same shared model tag for every patient, "
                        "for example trpo_pop_fixdt.")
    p.add_argument("--method_name", type=str, default=None,
                   help="Short method slug used in saved evaluation dataset names. "
                        "If omitted, inferred from --save_dir or --cql_model_dir, "
                        "for example cql, dqn, or gcql.")
    p.add_argument("--dt_modes", nargs="*", default=["fixdt"], choices=["fixdt", "vardt"])
    p.add_argument("--eval_scope", type=str, default="both", choices=["individual", "population_transfer", "both"],
                    help=(
                        "Which offline policy scope to evaluate. "
                        "'individual' evaluates per-patient models pid_<pid>_fixdt; "
                        "'population_transfer' evaluates a shared model such as trpo_pop_fixdt; "
                        "'both' evaluates both."
                    ),)
    p.add_argument("--n_eval",        type=int,   default=5)
    p.add_argument("--K",             type=int,   default=20)
    p.add_argument("--hidden",        type=int,   default=256)
    p.add_argument("--action_noise_std", type=float, default=0.05)
    p.add_argument("--patient_ids",   type=int,   nargs="*",
                   default=None)#[200325, 201046, 201101, 202028, 202515])
    p.add_argument("--patient_ids_file", type=str, default=None,
        help="Optional pid-only CSV/TXT file, e.g. checkpoints-cohort/cohort_1/cohort_1_training.csv. "
            "If provided, it is used instead of --patient_ids for evaluation.",)
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
        discovered_pids = discover_patient_ids_from_pinn_dir(pinn_base)

        if requested_pids is not None:
            available = set(discovered_pids)
            missing = [p for p in requested_pids if p not in available]
            if missing:
                print(
                    f"[warning] {len(missing)} requested patients do not have PINNs under "
                    f"{args.pinn_dir}; first missing IDs: {missing[:10]}",
                    flush=True,
                )
            # Preserve the order in the patient list / CLI input.
            pids = [p for p in requested_pids if p in available]
        else:
            pids = discovered_pids

    else:
        from data.loader import load_all_patients

        bundle = load_all_patients(args.csv)
        all_pats = sorted(
            [
                p for p in bundle.patients
                if (pinn_base / f"patient_{p['icu_id']}" / "pinn.pt").exists()
            ],
            key=lambda p: p["icu_id"],
        )

        available = {p["icu_id"] for p in all_pats}

        if requested_pids is not None:
            missing = [p for p in requested_pids if p not in available]
            if missing:
                print(
                    f"[warning] {len(missing)} requested patients are not available "
                    f"from the CSV/PINN folders; first missing IDs: {missing[:10]}",
                    flush=True,
                )
            # Preserve the order in the patient list / CLI input.
            pids = [p for p in requested_pids if p in available]
        else:
            pids = [p["icu_id"] for p in all_pats]

    if not pids:
        raise RuntimeError(
            f"No patient PINNs found under {args.pinn_dir} for the requested patients. "
            "Check --patient_ids_file, --patient_ids, and the PINN folder layout."
        )

    per_patient_dir = os.path.join(args.save_dir, "per_patient")
    summary_dir = os.path.join(args.save_dir, "summary")
    dataset_dir = os.path.join(args.save_dir, "datasets")
    os.makedirs(per_patient_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    glb_scales_path = resolve_scales_path(args.glb_scales)
    csv_arg = None if args.no_mimic_csv else args.csv
    method_name = _infer_method_name(args.save_dir, args.cql_model_dir, args.method_name)

    if args.eval_scope == "individual":
        active_policies = ["ind_fixdt"]
    elif args.eval_scope == "population_transfer":
        active_policies = ["glb_fixdt"]
    else:
        active_policies = ["ind_fixdt", "glb_fixdt"]

    # Keep only policies corresponding to requested dt modes.
    requested_tags = set(args.dt_modes)
    active_policies = [
        p for p in active_policies
        if ("fixdt" in p and "fixdt" in requested_tags)
        or ("vardt" in p and "vardt" in requested_tags)
    ]

    print(f"\n{'=' * 60}", flush=True)
    print(f"  eval_offline", flush=True)
    print(f"  pinn_dir         : {args.pinn_dir}", flush=True)
    print(f"  no_mimic_csv     : {args.no_mimic_csv}", flush=True)
    print(f"  patient_ids_file : {args.patient_ids_file}", flush=True)
    print(f"  eval_scope       : {args.eval_scope}", flush=True)
    print(f"  cql_model_dir    : {args.cql_model_dir}", flush=True)
    print(f"  glb_scales       : {glb_scales_path}", flush=True)
    print(f"  model_tag        : {args.model_tag}", flush=True)
    print(f"  method_name      : {method_name}", flush=True)
    print(f"  save_dir         : {args.save_dir}", flush=True)
    print(f"  n_eval           : {args.n_eval}  K={args.K}", flush=True)
    print(f"  hidden           : {args.hidden}", flush=True)
    print(f"  noise_std        : {args.action_noise_std}", flush=True)
    print(f"  dt_modes         : {' '.join(args.dt_modes)}", flush=True)
    print(f"  patients         : {len(pids)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    all_results = {}
    glb_datasets_per_patient = {}

    for pid in pids:
        ret = _worker(
            pid,
            args.pinn_dir,
            args.cql_model_dir,
            per_patient_dir,
            csv_arg,
            args.n_eval,
            args.K,
            args.hidden,
            args.action_noise_std,
            dataset_dir,
            glb_scales_path=glb_scales_path,
            model_tag=args.model_tag,
            dt_modes=args.dt_modes,
            eval_scope=args.eval_scope,
            method_name=method_name,
        )
        if ret is None:
            continue

        results, glb_ds = ret
        all_results[pid] = results
        glb_datasets_per_patient[pid] = glb_ds
        _plot_per_patient(pid, results, per_patient_dir, active_policies=active_policies)

    if not all_results:
        raise RuntimeError(
            "No valid evaluation results were produced. Check patient PINNs and model directories."
        )

    _plot_summary(all_results, summary_dir, active_policies=active_policies)

    if args.eval_scope in ("population_transfer", "both"):
        _merge_and_save_global(
            glb_datasets_per_patient,
            dataset_dir,
            args.action_noise_std,
            method_name=method_name,
            dt_modes=args.dt_modes,
        )

    print(f"\nDone.  Plots    → {args.save_dir}", flush=True)
    print(f"       Datasets → {dataset_dir}", flush=True)

if __name__ == "__main__":
    main()