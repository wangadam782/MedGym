"""Evaluate Ind / Cluster-pooled / Pop policies with the old K-based workflow.

Evaluates Ind / Cluster-pooled / Pop policies for one algorithm.  It keeps the
same command shape, for example:

  python scripts/online/eval_ind_cluster_pop.py --algo sac --K 20
  python scripts/online/eval_ind_cluster_pop.py --algo lagrangian_trpo --K 20

Additional cluster-pooled inputs:

  --cluster_root        cluster policy root
  --cluster_scales_root cluster PINN/scales root
  --cluster_map         CSV mapping patient_id -> cluster_id
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import _eval_base as base

from data.config import (
    action_dim,
    state_dim,
    PINN_IND_ROOT,
    POLICY_IND_ROOT,
    POLICY_POP_ROOT,
    PINN_POP_ROOT,
    POLICY_TRANSFER_EVAL_ROOT,
)


COLOR_IND = base.COLOR_IND
COLOR_POP = base.COLOR_POP
COLOR_CLUSTER = "#2CA02C"
SOFA_LEGEND_FONTSIZE = 18
BETTER_THAN_POP_MARKER = "^"
WORSE_THAN_POP_MARKER = "x"
COMPARISON_MARKER_SIZE = 12

DEFAULT_CLUSTER_MAP = (
    _ROOT / "reproduce" / "extra110_cohort_7_train_only" / "cluster_map.csv"
)
DEFAULT_CLUSTER_SCALES_ROOT = _ROOT / "results" / "pinn" / "cluster_pooled"
DEFAULT_CLUSTER_ROOT = _ROOT / "results" / "online" / "cluster_pooled"


def _add_population_comparison_markers(
    ax: plt.Axes,
    xs: np.ndarray,
    policy_vals: np.ndarray,
    population_vals: np.ndarray,
    color: str,
) -> tuple[float, float]:
    valid = np.isfinite(policy_vals) & np.isfinite(population_vals)
    better = valid & (policy_vals < population_vals)
    worse = valid & (policy_vals > population_vals)
    n_valid = int(np.sum(valid))
    lower_pct = float(np.sum(better) / n_valid * 100.0) if n_valid else 0.0
    higher_pct = float(np.sum(worse) / n_valid * 100.0) if n_valid else 0.0
    if np.any(better):
        ax.plot(
            xs[better],
            policy_vals[better],
            linestyle="None",
            marker=BETTER_THAN_POP_MARKER,
            ms=COMPARISON_MARKER_SIZE,
            color=color,
            mec="#333333",
            mew=0.8,
            zorder=4,
        )
    if np.any(worse):
        ax.plot(
            xs[worse],
            policy_vals[worse],
            linestyle="None",
            marker=WORSE_THAN_POP_MARKER,
            ms=COMPARISON_MARKER_SIZE,
            color=color,
            mew=1.9,
            zorder=4,
        )
    return lower_pct, higher_pct


def _comparison_legend_handles(color: str) -> tuple[Line2D, Line2D]:
    lower_handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker=BETTER_THAN_POP_MARKER,
        ms=COMPARISON_MARKER_SIZE,
        color=color,
        mec="#333333",
        mew=0.8,
    )
    higher_handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker=WORSE_THAN_POP_MARKER,
        ms=COMPARISON_MARKER_SIZE,
        color=color,
        mew=1.9,
    )
    return lower_handle, higher_handle


def _policies_for(algo: str) -> list[str]:
    return [
        f"{algo}_ind_fixdt",
        f"{algo}_cluster_pooled_fixdt",
        f"{algo}_pop_fixdt",
        f"{algo}_ind_vardt",
        f"{algo}_cluster_pooled_vardt",
        f"{algo}_pop_vardt",
    ]


def _labels_for(algo: str) -> dict[str, str]:
    pretty = base.ALGO_REG[algo]["label"]
    return {
        f"{algo}_ind_fixdt": f"{pretty}-F-Ind",
        f"{algo}_cluster_pooled_fixdt": f"{pretty}-F-Clu",
        f"{algo}_pop_fixdt": f"{pretty}-F-Pop",
        f"{algo}_ind_vardt": f"{pretty}-A-Ind",
        f"{algo}_cluster_pooled_vardt": f"{pretty}-A-Clu",
        f"{algo}_pop_vardt": f"{pretty}-A-Pop",
    }


def _colors_for(algo: str) -> dict[str, str]:
    return {
        f"{algo}_ind_fixdt": COLOR_IND,
        f"{algo}_cluster_pooled_fixdt": COLOR_CLUSTER,
        f"{algo}_pop_fixdt": COLOR_POP,
        f"{algo}_ind_vardt": COLOR_IND,
        f"{algo}_cluster_pooled_vardt": COLOR_CLUSTER,
        f"{algo}_pop_vardt": COLOR_POP,
    }


def _lighten_toward_white(hex_or_rgb, frac: float = 0.58) -> tuple:
    r, g, b = mcolors.to_rgb(hex_or_rgb)
    return tuple(c + (1.0 - c) * frac for c in (r, g, b))


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_cluster_map(path: Path) -> dict[int, int]:
    if not path.exists():
        return {}
    out: dict[int, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "patient_id" not in row or "cluster_id" not in row:
                continue
            try:
                out[int(row["patient_id"])] = int(row["cluster_id"])
            except (TypeError, ValueError):
                continue
    return out


def _load_scales_any(path: Path):
    raw = np.load(path, allow_pickle=True)
    if hasattr(raw, "files"):
        keys = set(raw.files)
        if {"state_mean", "state_std", "action_scale"}.issubset(keys):
            mean = raw["state_mean"]
            std = raw["state_std"]
            ascl = raw["action_scale"]
        elif {"mean", "std", "ascl"}.issubset(keys):
            mean = raw["mean"]
            std = raw["std"]
            ascl = raw["ascl"]
        else:
            raise KeyError(f"Unsupported scale keys in {path}: {sorted(keys)}")
    else:
        d = raw.item()
        mean = d.get("state_mean", d.get("mean"))
        std = d.get("state_std", d.get("std"))
        ascl = d.get("action_scale", d.get("ascl"))
        if mean is None or std is None or ascl is None:
            raise KeyError(f"Unsupported scale dict keys in {path}: {sorted(d.keys())}")
    return (
        np.asarray(mean, dtype=np.float32),
        np.asarray(std, dtype=np.float32),
        np.asarray(ascl, dtype=np.float32),
    )


def _resolve_cluster_scales(cluster_scales_root: Path, cluster_id: int) -> Path | None:
    return _first_existing(
        cluster_scales_root / f"cluster_{cluster_id}" / "scales.npz",
        cluster_scales_root / f"cluster_{cluster_id}" / "cluster_scales.npz",
        cluster_scales_root / f"cluster_{cluster_id}" / "scales.npy",
        cluster_scales_root / f"cluster_{cluster_id}" / "cluster_scales.npy",
        cluster_scales_root / f"cluster_{cluster_id}_scales.npz",
        cluster_scales_root / f"cluster_{cluster_id}_scales.npy",
        # remaining15 layout: clu_<nearest_pid>/
        cluster_scales_root / f"clu_{cluster_id}" / "scales.npy",
        cluster_scales_root / f"clu_{cluster_id}" / "scales.npz",
    )


def _resolve_cluster_path(
    cluster_root: Path,
    cluster_id: int,
    tag: str,
    K: int,
    file_prefix: str,
) -> Path | None:
    """Resolve cluster-pooled policy path across old and full110 layouts."""
    cid = str(cluster_id)
    cluster_dir = f"cluster_{cid}"
    clu_dir     = f"clu_{cid}"        # remaining15 layout: clu_<nearest_pid>/
    policy_dir = f"cluster_pooled_cluster_{cid}_pooled"
    candidates = [
        # Old K-based layouts.
        cluster_root / cluster_dir / file_prefix / f"K{K}" / tag / "policy.pt",
        cluster_root / cluster_dir / file_prefix / f"K{K}" / tag / "actor.pt",
        cluster_root / file_prefix / f"K{K}" / tag / cluster_dir / "policy.pt",
        cluster_root / file_prefix / f"K{K}" / tag / cluster_dir / "actor.pt",
        cluster_root / cluster_dir / tag / f"{file_prefix}_{tag}.pt",
        cluster_root / cluster_dir / f"{file_prefix}_{tag}_K{K}" / f"{file_prefix}_{tag}.pt",
        # remaining15 layout.
        cluster_root / clu_dir / file_prefix / f"K{K}" / tag / "policy.pt",
        cluster_root / clu_dir / file_prefix / f"K{K}" / tag / "actor.pt",
        cluster_root / clu_dir / tag / f"{file_prefix}_{tag}.pt",
        # full110 runner layouts if the caller explicitly points here.
        cluster_root / file_prefix / cluster_dir / policy_dir / f"{file_prefix}_best.pt",
        cluster_root / file_prefix / cluster_dir / policy_dir / f"{file_prefix}_last.pt",
        cluster_root / cluster_dir / policy_dir / f"{file_prefix}_best.pt",
        cluster_root / cluster_dir / policy_dir / f"{file_prefix}_last.pt",
    ]
    return _first_existing(*candidates)


def _build_agent(agent_cls, pt_path: Path, default_hidden: int):
    import torch
    from rl import OPTION_ACTION_DIM, SMDP_STATE_DIM
    from data.config import device

    raw = torch.load(str(pt_path), map_location=device, weights_only=False)
    actor_sd = base._extract_actor_sd(raw)
    hidden_use = base._infer_hidden(actor_sd) or default_hidden
    action_dim_use = base._infer_action_dim(actor_sd) or OPTION_ACTION_DIM
    agent = agent_cls(state_dim=SMDP_STATE_DIM, action_dim=action_dim_use, hidden=hidden_use)
    try:
        agent.actor.load_state_dict(actor_sd)
    except Exception:
        agent.load(str(pt_path))
    return agent


def _worker(
    pid: int,
    algo: str,
    pinn_dir: str,
    ind_root: str,
    pop_root: str,
    glb_scales_path: str,
    cluster_root: str,
    cluster_scales_root: str,
    cluster_map: dict[int, int],
    n_eval: int,
    K: int,
    init_noise_std: float = 0.0,
    eval_seed: int = 42,
):
    import torch
    import rl as _rl
    from models import PINN
    from rl import ICUEnvironment
    from data.config import device as global_device

    device = global_device
    cfg = base.ALGO_REG[algo]
    agent_cls = getattr(_rl, cfg["cls_name"])
    file_prefix = cfg["file_prefix"]
    default_hidden = cfg["default_hidden"]

    pinn_path = base._resolve_pinn_path(Path(pinn_dir), pid)
    if pinn_path is None:
        print(f"[patient {pid}] PINN not found under {pinn_dir}, skip.", flush=True)
        return None

    try:
        pat_mean, pat_std, pat_ascl, state_min_np, state_max_np, _ = base.load_pinn_folder_scales(
            pinn_path.parent,
            base.PHYSICAL_MIN,
            base.PHYSICAL_MAX,
        )
    except FileNotFoundError as exc:
        print(f"[patient {pid}] {exc}, skipping.", flush=True)
        return None

    init_norm_pat = base.load_individual_init_state_norm(
        pinn_path.parent,
        state_dim=state_dim,
        state_min_np=state_min_np,
        state_max_np=state_max_np,
    )
    if init_norm_pat is None:
        print(f"[patient {pid}] missing init_state_norm.npy under {pinn_path.parent}; skip.", flush=True)
        return None
    init_norm_pat = np.asarray(init_norm_pat, dtype=np.float32)

    pinn = PINN(state_dim, action_dim).to(device)
    pinn.load_state_dict(torch.load(str(pinn_path), map_location=device))
    pinn.eval()

    fixed_dt = base.TOTAL_TIME_H / K

    def make_patient_env(use_dt: bool):
        return ICUEnvironment(
            pinn_model=pinn,
            mean_np=pat_mean,
            std_np=pat_std,
            action_min_norm=np.zeros(action_dim, dtype=np.float32),
            action_max_norm=np.ones(action_dim, dtype=np.float32),
            state_min=state_min_np,
            state_max=state_max_np,
            action_scale_np=pat_ascl,
            max_steps=K,
            dt_min=0.5,
            dt_max=36.0,
            total_time_h=base.TOTAL_TIME_H,
            use_dt=use_dt,
            fixed_dt=fixed_dt,
            use_lac_penalty=True,
        )

    rng = np.random.default_rng(eval_seed)
    results = {key: None for key in _policies_for(algo)}

    def _store(key: str, agent, env):
        sofa_mean, sofa_std, lac_mean, lac_std, m, sofa_mat, lac_mat, per_ep = base._collect_policy(
            agent,
            env,
            n_eval,
            init_norm_pat=init_norm_pat,
            init_noise_std=init_noise_std,
            rng=rng,
        )
        results[key] = {
            "sofa_mean": sofa_mean,
            "sofa_std": sofa_std,
            "lac_mean": lac_mean,
            "lac_std": lac_std,
            "sofa_mat": sofa_mat,
            "lac_mat": lac_mat,
            "per_ep": per_ep,
            **m,
        }
        print(
            f"[patient {pid}] {key:<36} "
            f"sofa20={m['mean_sofa']:.2f} sofa96={m['final_sofa']:.2f} "
            f"lac={m['mean_lac']:.2f} safe={m['safety_rate']:.2%} dt={m['mean_dt']:.2f}h",
            flush=True,
        )

    # Individual policies.
    for tag, use_dt in [("fixdt", False), ("vardt", True)]:
        key = f"{algo}_ind_{tag}"
        pt_path = base._resolve_ind_path(Path(ind_root), pid, tag, K, file_prefix)
        if pt_path is None:
            print(f"[patient {pid}] {key}: model not found.", flush=True)
            continue
        agent = _build_agent(agent_cls, pt_path, default_hidden)
        _store(key, agent, make_patient_env(use_dt))

    # Population policies.
    pop_scales = None
    glb_path = Path(glb_scales_path)
    if glb_path.exists():
        raw = np.load(glb_path, allow_pickle=True).item()
        pop_scales = (
            raw["mean"].astype(np.float32),
            raw["std"].astype(np.float32),
            raw["ascl"].astype(np.float32),
        )
    else:
        print(f"[patient {pid}] population scales not found: {glb_path}", flush=True)

    if pop_scales is not None:
        gl_mean, gl_std, gl_ascl = pop_scales
        for tag, use_dt in [("fixdt", False), ("vardt", True)]:
            key = f"{algo}_pop_{tag}"
            pt_path = base._resolve_pop_path(Path(pop_root), tag, K, file_prefix)
            if pt_path is None:
                print(f"[patient {pid}] {key}: model not found.", flush=True)
                continue
            agent = _build_agent(agent_cls, pt_path, default_hidden)
            env = base.AdaptedEnv(
                make_patient_env(use_dt),
                pat_mean,
                pat_std,
                pat_ascl,
                gl_mean,
                gl_std,
                gl_ascl,
            )
            _store(key, agent, env)

    # Cluster-pooled policies.
    cluster_id = cluster_map.get(int(pid))
    if cluster_id is None:
        print(f"[patient {pid}] cluster id not found in cluster_map; cluster skipped.", flush=True)
        return results

    cluster_scale_path = _resolve_cluster_scales(Path(cluster_scales_root), cluster_id)
    if cluster_scale_path is None:
        print(f"[patient {pid}] cluster {cluster_id}: scales not found; cluster skipped.", flush=True)
        return results
    clu_mean, clu_std, clu_ascl = _load_scales_any(cluster_scale_path)

    for tag, use_dt in [("fixdt", False), ("vardt", True)]:
        key = f"{algo}_cluster_pooled_{tag}"
        pt_path = _resolve_cluster_path(Path(cluster_root), cluster_id, tag, K, file_prefix)
        if pt_path is None:
            print(f"[patient {pid}] {key}: model not found for cluster {cluster_id}.", flush=True)
            continue
        agent = _build_agent(agent_cls, pt_path, default_hidden)
        env = base.AdaptedEnv(
            make_patient_env(use_dt),
            pat_mean,
            pat_std,
            pat_ascl,
            clu_mean,
            clu_std,
            clu_ascl,
        )
        _store(key, agent, env)

    return results


def _save_trajectories(pid: int, results: dict, save_dir: str, algo: str) -> None:
    pid_dir = os.path.join(save_dir, str(pid))
    os.makedirs(pid_dir, exist_ok=True)
    for key in _policies_for(algo):
        r = results.get(key)
        if r is None or "sofa_mat" not in r:
            continue
        per_ep = r.get("per_ep", {})
        np.savez(
            os.path.join(pid_dir, f"rollout_{key}.npz"),
            t_grid=base.T_GRID.astype(np.float32),
            sofa_mat=r["sofa_mat"].astype(np.float32),
            lac_mat=r["lac_mat"].astype(np.float32),
            sofa_mean=r["sofa_mean"].astype(np.float32),
            sofa_std=r["sofa_std"].astype(np.float32),
            lac_mean=r["lac_mean"].astype(np.float32),
            lac_std=r["lac_std"].astype(np.float32),
            **{f"per_ep_{k}": v for k, v in per_ep.items()},
        )


def _plot_per_patient(pid: int, results: dict, save_dir: str, algo: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    labels = _labels_for(algo)
    colors = _colors_for(algo)

    for tag, title in [("fixdt", "fixed interval time"), ("vardt", "adaptive interval time")]:
        keys = [
            f"{algo}_ind_{tag}",
            f"{algo}_cluster_pooled_{tag}",
            f"{algo}_pop_{tag}",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        ax = axes[0]
        for key in keys:
            if results.get(key) is None:
                continue
            mt = results[key]["sofa_mean"]
            st = results[key]["sofa_std"]
            c = colors[key]
            ax.plot(base.T_GRID, mt, color=c, lw=2.0, label=labels[key])
            ax.fill_between(base.T_GRID, mt - st, mt + st, color=_lighten_toward_white(c), alpha=0.30)
        ax.axvline(base.POST_HOUR, color="gray", ls="--", lw=1.0)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("SOFA score")
        ax.set_title(f"SOFA - {title}")
        ax.legend(fontsize=8)

        ax = axes[1]
        for key in keys:
            if results.get(key) is None:
                continue
            mt = results[key]["lac_mean"]
            st = results[key]["lac_std"]
            c = colors[key]
            ax.plot(base.T_GRID, mt, color=c, lw=2.0, label=labels[key])
            ax.fill_between(base.T_GRID, mt - st, mt + st, color=_lighten_toward_white(c), alpha=0.30)
        ax.axhline(base.LAC_DANGER_THR, color="#4A148C", ls="--", lw=1.2)
        ax.axhline(base.LAC_NORMAL_THR, color="darkorange", ls=":", lw=1.0)
        ax.axvline(base.POST_HOUR, color="gray", ls="--", lw=1.0)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Lactate (mmol/L)")
        ax.set_title(f"Lactate - {title}")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sofa_lactate_pid{pid}_{tag}.png"), dpi=150)
        plt.close(fig)


def _plot_summary(
    all_results: dict,
    save_dir: str,
    algo: str,
    K: int = 20,
    cluster_map: dict[int, int] | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    policies = _policies_for(algo)
    labels = _labels_for(algo)
    colors = _colors_for(algo)
    fname_prefix = f"{algo}_K{K}_"
    if cluster_map:
        pids = sorted(all_results.keys(), key=lambda pid: (cluster_map.get(int(pid), 10**9), int(pid)))
    else:
        pids = sorted(all_results.keys())
    cluster_ids = [cluster_map.get(int(pid)) if cluster_map else None for pid in pids]
    cluster_tick_labels = [
        str(cid) if cid is not None else str(pid)
        for pid, cid in zip(pids, cluster_ids)
    ]

    def _extr(key: str, metric: str):
        return np.array([
            all_results[p][key][metric] if all_results[p].get(key) is not None else np.nan
            for p in pids
        ], dtype=float)

    metrics_table = {}
    for key in policies:
        metrics_table[key] = {
            "sofa20": _extr(key, "mean_sofa"),
            "sofa_all": _extr(key, "mean_sofa_all"),
            "sofa96": _extr(key, "final_sofa"),
            "mean_lac": _extr(key, "mean_lac"),
            "safety": _extr(key, "safety_rate"),
            "clr6h": _extr(key, "clearance_6h"),
            "tabove4": _extr(key, "time_above_4"),
            "mean_dt": _extr(key, "mean_dt"),
        }

    def _strip(v):
        return v[np.isfinite(v)]

    def _ylim(metric_name: str, pad: float = 0.5):
        vals = [_strip(metrics_table[k][metric_name]) for k in policies]
        vals = [v for v in vals if len(v) > 0]
        if not vals:
            return 0.0, 1.0
        all_vals = np.concatenate(vals)
        return float(np.nanmin(all_vals)) - pad, float(np.nanmax(all_vals)) + pad

    xs = np.arange(len(pids))
    axis_fs = 30
    tick_fs = 20
    legend_fs = 20
    sofa_axis_fs = 36
    sofa_tick_fs = 24
    sofa_legend_fs = SOFA_LEGEND_FONTSIZE
    sofa_legend_kwargs = {
        "fontsize": sofa_legend_fs,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.12),
        "ncol": 3,
        "framealpha": 0.92,
        "borderaxespad": 0.0,
        "handlelength": 1.4,
        "handletextpad": 0.4,
        "columnspacing": 0.8,
    }
    sofa_alltime_dist_legend_kwargs = {
        "fontsize": 18,
        "loc": "upper right",
        "bbox_to_anchor": (0.98, 0.98),
        "ncol": 1,
        "framealpha": 0.88,
        "borderaxespad": 0.0,
        "handlelength": 1.4,
        "handletextpad": 0.4,
        "labelspacing": 0.35,
    }
    def _line(
        fname: str,
        metric: str,
        ylabel: str,
        ylim,
        tag: str,
        safety_scale: float = 1.0,
        x_tick_labels: list[str] | None = None,
        x_axis_label: str = "Patient (sorted by ID)",
    ):
        keys = [
            f"{algo}_ind_{tag}",
            f"{algo}_cluster_pooled_{tag}",
            f"{algo}_pop_{tag}",
        ]
        markers = ["o", "^", "s"]
        use_cluster_pop_markers = metric in {"sofa_all", "sofa96"}
        fig, ax = plt.subplots(figsize=(12, 6))
        pop_key = f"{algo}_pop_{tag}"
        series_by_key: dict[str, np.ndarray] = {}
        line_handles = []
        line_labels = []
        for key, marker in zip(keys, markers):
            vals = metrics_table[key][metric] * safety_scale
            series_by_key[key] = np.asarray(vals, dtype=float)
            label = labels[key]
            line, = ax.plot(
                xs,
                vals,
                color=colors[key],
                lw=2.2,
                marker=None if use_cluster_pop_markers else marker,
                ms=7,
                mec="#333333",
                mew=0.8,
            )
            line_handles.append(line)
            line_labels.append(label)
        if use_cluster_pop_markers:
            cluster_key = f"{algo}_cluster_pooled_{tag}"
            ind_key = f"{algo}_ind_{tag}"
            cluster_lower_pct, cluster_higher_pct = _add_population_comparison_markers(
                ax,
                xs,
                series_by_key[cluster_key],
                series_by_key[pop_key],
                colors[cluster_key],
            )
            ind_lower_pct, ind_higher_pct = _add_population_comparison_markers(
                ax,
                xs,
                series_by_key[ind_key],
                series_by_key[pop_key],
                colors[ind_key],
            )
            cluster_lower_handle, cluster_higher_handle = _comparison_legend_handles(colors[cluster_key])
            ind_lower_handle, ind_higher_handle = _comparison_legend_handles(colors[ind_key])
            marker_handles = [
                cluster_lower_handle,
                cluster_higher_handle,
                ind_lower_handle,
                ind_higher_handle,
            ]
            marker_labels = [
                f"Clu Better: {cluster_lower_pct:.0f}%",
                f"Clu Worse: {cluster_higher_pct:.0f}%",
                f"Ind Better: {ind_lower_pct:.0f}%",
                f"Ind Worse: {ind_higher_pct:.0f}%",
            ]
        use_sofa_style = metric in {"sofa20", "sofa_all", "sofa96"}
        ax.set_xlabel(x_axis_label, fontsize=sofa_axis_fs if use_sofa_style else axis_fs)
        ax.set_ylabel(ylabel, fontsize=sofa_axis_fs if use_sofa_style else axis_fs)
        if x_tick_labels is None:
            ax.set_xticks([])
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(x_tick_labels, rotation=0, ha="center")
        if ylim is not None:
            ax.set_ylim(*ylim)
        bbox_extra_artists = ()
        if use_sofa_style:
            legend_kwargs = dict(sofa_legend_kwargs)
            if use_cluster_pop_markers:
                legend_kwargs["ncol"] = 3
                main_legend = ax.legend(line_handles, line_labels, **legend_kwargs)
                ax.add_artist(main_legend)
                bbox_extra_artists = (main_legend,)
                ax.legend(
                    marker_handles,
                    marker_labels,
                    fontsize=sofa_legend_fs,
                    loc="upper right",
                    ncol=2,
                    framealpha=0.92,
                    borderaxespad=0.3,
                    handlelength=1.0,
                    handletextpad=0.35,
                    columnspacing=0.8,
                    labelspacing=0.25,
                )
            else:
                ax.legend(line_handles, line_labels, **legend_kwargs)
        else:
            ax.legend(line_handles, line_labels, fontsize=legend_fs, loc="best")
        ax.tick_params(labelsize=sofa_tick_fs if use_sofa_style else tick_fs)
        plt.tight_layout()
        savefig_kwargs = {
            "dpi": 150,
            "bbox_inches": "tight",
            "pad_inches": 0.25,
        }
        if bbox_extra_artists:
            savefig_kwargs["bbox_extra_artists"] = bbox_extra_artists
        plt.savefig(os.path.join(save_dir, fname_prefix + fname), **savefig_kwargs)
        plt.close(fig)

    for tag in ("fixdt", "vardt"):
        _line(
            f"mean_sofa_per_patient_{tag}.png",
            "sofa20",
            "Mean SOFA (t >= 20 h)",
            _ylim("sofa20"),
            tag,
        )
        _line(
            f"mean_sofa_alltime_per_patient_{tag}.png",
            "sofa_all",
            "Mean SOFA (all time)",
            _ylim("sofa_all"),
            tag,
            x_tick_labels=cluster_tick_labels,
            x_axis_label="Cluster",
        )
        _line(
            f"final_sofa_per_patient_{tag}.png",
            "sofa96",
            "Final SOFA (t = 96 h)",
            _ylim("sofa96"),
            tag,
            x_tick_labels=cluster_tick_labels,
            x_axis_label="Cluster",
        )
        _line(
            f"safety_rate_{tag}.png",
            "safety",
            f"Safety rate (lac < {base.LAC_DANGER_THR:.0f}) [%]",
            (0, 105),
            tag,
            safety_scale=100.0,
        )

    try:
        from scipy.stats import gaussian_kde
        kde_ok = True
    except Exception:
        gaussian_kde = None
        kde_ok = False

    def _win_rate_lower_is_better(v_left, v_right) -> float:
        left = np.asarray(v_left, dtype=float)
        right = np.asarray(v_right, dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        if not np.any(valid):
            return 0.0
        return float(np.sum(left[valid] < right[valid]) / np.sum(valid) * 100.0)

    def _density_curve(values: np.ndarray, xlims: tuple[float, float]):
        vals = _strip(np.asarray(values, dtype=float))
        if vals.size == 0:
            return None
        grid = np.linspace(xlims[0], xlims[1], 400)
        if kde_ok and vals.size >= 2:
            try:
                kde = gaussian_kde(vals)
                return grid, kde(grid)
            except Exception as exc:
                print(f"[KDE] fallback density: {exc}", flush=True)
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if not np.isfinite(sigma) or sigma < 1e-6:
            span = max(float(xlims[1] - xlims[0]), 1.0)
            sigma = span / 100.0
        dens = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        return grid, dens

    def _sofa_dist(fname: str, metric: str, xlabel: str, xlims, tag: str):
        keys = [
            f"{algo}_ind_{tag}",
            f"{algo}_cluster_pooled_{tag}",
            f"{algo}_pop_{tag}",
        ]
        pop_key = f"{algo}_pop_{tag}"
        pop_vals = np.asarray(metrics_table[pop_key][metric], dtype=float)
        fig, ax = plt.subplots(figsize=(11, 6))
        plotted = 0
        for key in keys:
            vals = _strip(np.asarray(metrics_table[key][metric], dtype=float))
            curve = _density_curve(vals, xlims)
            if curve is None:
                continue
            grid, dens = curve
            c = colors[key]
            label = labels[key]
            if key != pop_key:
                lower_pct = _win_rate_lower_is_better(
                    np.asarray(metrics_table[key][metric], dtype=float),
                    pop_vals,
                )
                label = f"{label}  Better:{lower_pct:.0f}%"
            ax.fill_between(grid, dens, color=c, alpha=0.16)
            ax.plot(grid, dens, color=c, lw=2.5, label=label)
            ax.axvline(float(np.mean(vals)), color=c, ls="--", lw=1.4, alpha=0.85)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            return
        ax.set_xlabel(xlabel, fontsize=sofa_axis_fs)
        ax.set_ylabel("Probability density", fontsize=sofa_axis_fs)
        ax.set_xlim(*xlims)
        ax.set_ylim(bottom=0)
        if metric == "sofa_all":
            ax.legend(**sofa_alltime_dist_legend_kwargs)
        else:
            ax.legend(**sofa_legend_kwargs)
        ax.tick_params(labelsize=sofa_tick_fs)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname_prefix + fname), dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    for tag in ("fixdt", "vardt"):
        _sofa_dist(
            f"mean_sofa_dist_{tag}.png",
            "sofa20",
            "Mean SOFA (t >= 20 h)",
            _ylim("sofa20"),
            tag,
        )
        _sofa_dist(
            f"mean_sofa_alltime_dist_{tag}.png",
            "sofa_all",
            "Mean SOFA (all time)",
            _ylim("sofa_all"),
            tag,
        )
        _sofa_dist(
            f"final_sofa_dist_{tag}.png",
            "sofa96",
            "Final SOFA (t = 96 h)",
            _ylim("sofa96"),
            tag,
        )

    order = policies
    label_order = [labels[k] for k in order]
    color_order = [colors[k] for k in order]

    def _boxplot(fname: str, metric: str, ylabel: str, ylim, scale: float = 1.0):
        fig, ax = plt.subplots(figsize=(13, 6))
        data = [_strip(metrics_table[k][metric]) * scale for k in order]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=label_order)
        for patch, c, k in zip(bp["boxes"], color_order, order):
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
            patch.set_edgecolor("0.35")
            patch.set_linewidth(0.9)
            if k.endswith("vardt"):
                patch.set_hatch("//")
        ax.set_ylabel(ylabel, fontsize=axis_fs)
        ax.tick_params(axis="x", labelrotation=25, labelsize=14)
        ax.tick_params(axis="y", labelsize=tick_fs)
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname_prefix + fname), dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    _boxplot("boxplot_sofa_post20h.png", "sofa20", "Mean SOFA (t >= 20 h)", _ylim("sofa20"))
    _boxplot("boxplot_sofa_alltime.png", "sofa_all", "Mean SOFA (all time)", _ylim("sofa_all"))
    _boxplot("boxplot_sofa_final.png", "sofa96", "Final SOFA (t = 96 h)", _ylim("sofa96"))
    _boxplot(
        "boxplot_safety_rate.png",
        "safety",
        f"Safety rate (lac < {base.LAC_DANGER_THR:.0f}) [%]",
        (0, 105),
        scale=100.0,
    )

    metric_panels = [
        ("sofa20", "Mean SOFA (t >= 20 h)", _ylim("sofa20")),
        ("sofa_all", "Mean SOFA (all time)", _ylim("sofa_all")),
        ("sofa96", "Final SOFA", _ylim("sofa96")),
        ("safety", f"Safety rate [%]", (0, 115)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(25, 5.5))
    xs_bar = np.arange(len(order))
    for ax, (metric, ylabel, ylim) in zip(axes, metric_panels):
        scale = 100.0 if metric == "safety" else 1.0
        means = [np.nanmean(metrics_table[k][metric]) * scale for k in order]
        stds = [np.nanstd(metrics_table[k][metric]) * scale for k in order]
        ax.bar(xs_bar, means, yerr=stds, color=color_order, alpha=0.82, capsize=4)
        ax.set_xticks(xs_bar)
        ax.set_xticklabels(label_order, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname_prefix + "overall_stats.png"), dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    lines = [
        "policy,n,sofa_post20h_mean,sofa_post20h_std,sofa_alltime_mean,sofa_alltime_std,"
        "final_sofa_mean,final_sofa_std,safety_rate_mean,safety_rate_std,"
        "mean_lac_mean,mean_lac_std,clearance_6h_mean,time_above_4_mean,mean_dt"
    ]
    for key in order:
        sofa20 = metrics_table[key]["sofa20"]
        sofa_all = metrics_table[key]["sofa_all"]
        sofa96 = metrics_table[key]["sofa96"]
        safety = metrics_table[key]["safety"]
        mean_lac = metrics_table[key]["mean_lac"]
        clr6h = metrics_table[key]["clr6h"]
        tabove4 = metrics_table[key]["tabove4"]
        mean_dt = metrics_table[key]["mean_dt"]
        n = int(np.sum(np.isfinite(sofa20)))
        lines.append(
            f"{key},{n},"
            f"{np.nanmean(sofa20):.4f},{np.nanstd(sofa20):.4f},"
            f"{np.nanmean(sofa_all):.4f},{np.nanstd(sofa_all):.4f},"
            f"{np.nanmean(sofa96):.4f},{np.nanstd(sofa96):.4f},"
            f"{np.nanmean(safety):.4f},{np.nanstd(safety):.4f},"
            f"{np.nanmean(mean_lac):.4f},{np.nanstd(mean_lac):.4f},"
            f"{np.nanmean(clr6h):.4f},{np.nanmean(tabove4):.4f},{np.nanmean(mean_dt):.4f}"
        )
    with open(os.path.join(save_dir, "overall_stats.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    pp_lines = ["pid,cluster_id,policy,sofa20,sofa_all,sofa96,mean_lac,safety,clearance_6h,time_above_4,mean_dt"]
    for i, pid in enumerate(pids):
        cluster_id = "" if cluster_ids[i] is None else str(cluster_ids[i])
        for key in order:
            mt = metrics_table[key]
            pp_lines.append(
                f"{pid},{cluster_id},{key},"
                f"{mt['sofa20'][i]:.4f},{mt['sofa_all'][i]:.4f},"
                f"{mt['sofa96'][i]:.4f},{mt['mean_lac'][i]:.4f},"
                f"{mt['safety'][i]:.4f},{mt['clr6h'][i]:.4f},"
                f"{mt['tabove4'][i]:.4f},{mt['mean_dt'][i]:.4f}"
            )
    with open(os.path.join(save_dir, "per_patient_metrics.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(pp_lines) + "\n")

    winners = ["pid,cluster_id,dt_tag,winner,metric,best_value"]
    for i, pid in enumerate(pids):
        cluster_id = "" if cluster_ids[i] is None else str(cluster_ids[i])
        for tag in ("fixdt", "vardt"):
            keys = [f"{algo}_ind_{tag}", f"{algo}_cluster_pooled_{tag}", f"{algo}_pop_{tag}"]
            vals = [(k, metrics_table[k]["sofa20"][i]) for k in keys]
            vals = [(k, v) for k, v in vals if np.isfinite(v)]
            if not vals:
                continue
            best_key, best_val = min(vals, key=lambda item: item[1])
            winners.append(f"{pid},{cluster_id},{tag},{best_key},sofa20,{best_val:.4f}")
    with open(os.path.join(save_dir, "winner_by_patient.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(winners) + "\n")

    sofa_metrics = ["sofa20", "sofa_all", "sofa96"]

    def _safe_rate(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = (np.asarray(num, dtype=float) - np.asarray(den, dtype=float)) / np.asarray(den, dtype=float)
        return np.where(np.isfinite(rates), rates, np.nan)

    imp_cols = ["pid", "cluster_id"]
    imp_data: dict[str, np.ndarray] = {}
    for metric in sofa_metrics:
        for tag in ("fixdt", "vardt"):
            ind = metrics_table[f"{algo}_ind_{tag}"][metric]
            cluster = metrics_table[f"{algo}_cluster_pooled_{tag}"][metric]
            pop = metrics_table[f"{algo}_pop_{tag}"][metric]
            comparisons = {
                f"{metric}_pop2ind_{tag}": _safe_rate(ind, pop),
                f"{metric}_cluster2ind_{tag}": _safe_rate(ind, cluster),
                f"{metric}_pop2cluster_{tag}": _safe_rate(cluster, pop),
            }
            for col, values in comparisons.items():
                imp_cols.append(col)
                imp_data[col] = values
    for metric in sofa_metrics:
        for role in ("ind", "cluster_pooled", "pop"):
            col = f"{metric}_fix2var_{role}"
            imp_cols.append(col)
            imp_data[col] = _safe_rate(
                metrics_table[f"{algo}_{role}_vardt"][metric],
                metrics_table[f"{algo}_{role}_fixdt"][metric],
            )

    pp_imp_lines = [",".join(imp_cols)]
    for i, pid in enumerate(pids):
        row = [str(pid), "" if cluster_ids[i] is None else str(cluster_ids[i])]
        for col in imp_cols[2:]:
            value = imp_data[col][i]
            row.append(f"{value:.6f}" if np.isfinite(value) else "")
        pp_imp_lines.append(",".join(row))
    with open(os.path.join(save_dir, "per_patient_improvement.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(pp_imp_lines) + "\n")

    stat_lines = ["metric,comparison,mean,std,var,n"]
    for col in imp_cols[2:]:
        values = imp_data[col]
        finite = values[np.isfinite(values)]
        if "_pop2ind_" in col:
            metric, _, tail = col.partition("_pop2ind_")
            comparison = f"pop2ind_{tail}"
        elif "_cluster2ind_" in col:
            metric, _, tail = col.partition("_cluster2ind_")
            comparison = f"cluster2ind_{tail}"
        elif "_pop2cluster_" in col:
            metric, _, tail = col.partition("_pop2cluster_")
            comparison = f"pop2cluster_{tail}"
        elif "_fix2var_" in col:
            metric, _, tail = col.partition("_fix2var_")
            comparison = f"fix2var_{tail}"
        else:
            metric, comparison = col, ""
        if finite.size == 0:
            stat_lines.append(f"{metric},{comparison},nan,nan,nan,0")
            continue
        mean = float(finite.mean())
        std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        var = float(finite.var(ddof=1)) if finite.size > 1 else 0.0
        stat_lines.append(f"{metric},{comparison},{mean:.6f},{std:.6f},{var:.6f},{finite.size}")
    with open(os.path.join(save_dir, "improvement_stats.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(stat_lines) + "\n")

    np.save(
        os.path.join(save_dir, "summary_scores.npy"),
        {
            "algo": algo,
            "pids": pids,
            "cluster_ids": np.array([
                -1 if cluster_id is None else int(cluster_id)
                for cluster_id in cluster_ids
            ], dtype=np.int32),
            **{
                f"{key}_{metric}": metrics_table[key][metric]
                for key in policies
                for metric in ["sofa20", "sofa_all", "sofa96", "mean_lac", "safety", "clr6h", "tabove4", "mean_dt"]
            },
        },
    )


def _rebuild_all_results_from_summary_npy(path: str, algo: str) -> dict:
    raw = np.load(path, allow_pickle=True)
    blob = raw.item() if getattr(raw, "shape", None) == () else raw
    pids_order = [int(x) for x in blob["pids"]]
    metric_keys = [
        ("sofa20", "mean_sofa"),
        ("sofa_all", "mean_sofa_all"),
        ("sofa96", "final_sofa"),
        ("mean_lac", "mean_lac"),
        ("safety", "safety_rate"),
        ("clr6h", "clearance_6h"),
        ("tabove4", "time_above_4"),
        ("mean_dt", "mean_dt"),
    ]
    all_results: dict[int, dict] = {}
    for idx, pid in enumerate(pids_order):
        all_results[pid] = {}
        for policy in _policies_for(algo):
            s20_key = f"{policy}_sofa20"
            if s20_key not in blob:
                all_results[pid][policy] = None
                continue
            arr = np.asarray(blob[s20_key], dtype=float)
            if arr.size <= idx or not np.isfinite(arr.flat[idx]):
                all_results[pid][policy] = None
                continue
            entry = {}
            for stored_key, result_key in metric_keys:
                values = np.asarray(blob[f"{policy}_{stored_key}"], dtype=float)
                entry[result_key] = float(values.flat[idx]) if values.size > idx else float("nan")
            all_results[pid][policy] = entry
    return all_results


def _discover_default_patients(ind_root: Path, pinn_root: Path, file_prefix: str, K: int) -> list[int]:
    return base._discover_default_patients(ind_root, pinn_root, file_prefix, K)


def _save_meta(args, algo: str, pids: list[int], save_dir: Path) -> None:
    meta = {
        "script": "scripts/online/eval_ind_cluster_pop.py",
        "task": "transfer_eval_ind_cluster_pop",
        "algo_requested": args.algo,
        "algo": algo,
        "K": int(args.K),
        "n_eval": int(args.n_eval),
        "init_noise_std": float(args.init_noise_std),
        "eval_seed": int(args.eval_seed),
        "pinn_dir": str(Path(args.pinn_dir).resolve()),
        "ind_root": str(Path(args.ind_root).resolve()),
        "pop_root": str(Path(args.pop_root).resolve()),
        "cluster_root": str(Path(args.cluster_root).resolve()),
        "cluster_scales_root": str(Path(args.cluster_scales_root).resolve()),
        "cluster_map": str(Path(args.cluster_map).resolve()),
        "glb_scales": str(Path(args.glb_scales).resolve()),
        "save_dir": str(Path(save_dir).resolve()),
        "patient_ids": [int(x) for x in pids],
    }
    with open(Path(save_dir) / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _parse():
    p = argparse.ArgumentParser(
        description="Compare Ind / Cluster-pooled / Pop policies for one algorithm."
    )
    p.add_argument("--algo", required=True, choices=sorted(base.ALGO_REG.keys()))
    p.add_argument("--pinn_dir", default=PINN_IND_ROOT)
    p.add_argument("--ind_root", default=POLICY_IND_ROOT)
    p.add_argument("--pop_root", default=POLICY_POP_ROOT)
    p.add_argument("--cluster_root", default=str(DEFAULT_CLUSTER_ROOT))
    p.add_argument("--cluster_scales_root", default=str(DEFAULT_CLUSTER_SCALES_ROOT))
    p.add_argument("--cluster_map", default=str(DEFAULT_CLUSTER_MAP))
    p.add_argument("--glb_scales", default=str(Path(PINN_POP_ROOT) / "scales.npy"))
    p.add_argument("--save_root", default=str(POLICY_TRANSFER_EVAL_ROOT))
    p.add_argument("--save_dir", default=None)
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--init_noise_std", type=float, default=0.1)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--patient_ids", type=int, nargs="*", default=None)
    p.add_argument("--plot_only", action="store_true")
    return p.parse_args()


def main() -> None:
    def _noise_tag(x: float) -> str:
        return f"{x:g}".replace(".", "p")

    args = _parse()
    algo = args.algo
    cfg = base.ALGO_REG[algo]

    if args.save_dir is None:
        seed_tag = "" if args.eval_seed == 42 else f"_seed{args.eval_seed}"
        args.save_dir = (
            Path(args.save_root)
            / args.algo
            / f"K{args.K}"
            / f"ind_cluster_pop_n{args.n_eval}_noise{_noise_tag(args.init_noise_std)}{seed_tag}"
        )
    else:
        args.save_dir = Path(args.save_dir)

    summary_dir = os.path.join(args.save_dir, "summary")
    if args.plot_only:
        npy = os.path.join(summary_dir, "summary_scores.npy")
        if not os.path.isfile(npy):
            raise SystemExit(f"--plot_only: file not found: {npy}")
        all_results = _rebuild_all_results_from_summary_npy(npy, algo)
        cluster_map = _load_cluster_map(Path(args.cluster_map))
        _plot_summary(all_results, summary_dir, algo, args.K, cluster_map=cluster_map)
        print(f"[plot_only] updated -> {summary_dir}")
        return

    pinn_root = Path(args.pinn_dir)
    ind_root = Path(args.ind_root)
    if not pinn_root.exists():
        raise SystemExit(f"--pinn_dir not found: {pinn_root}")
    if not ind_root.exists():
        raise SystemExit(f"--ind_root not found: {ind_root}")

    if args.patient_ids:
        pids = list(args.patient_ids)
    else:
        pids = _discover_default_patients(ind_root, pinn_root, cfg["file_prefix"], args.K)
    if not pids:
        raise SystemExit("No patients to evaluate.")

    cluster_map = _load_cluster_map(Path(args.cluster_map))
    if not cluster_map:
        print(f"[WARN] cluster_map not found or empty: {args.cluster_map}", flush=True)

    per_patient_dir = os.path.join(args.save_dir, "per_patient")
    os.makedirs(per_patient_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    _save_meta(args, algo, pids, Path(args.save_dir))

    print(f"\n{'=' * 72}")
    print(f"  Transfer evaluation: {cfg['label']} Ind / Cluster / Pop")
    print(f"  algo             : {algo}")
    print(f"  K                : {args.K}")
    print(f"  pinn_dir         : {args.pinn_dir}")
    print(f"  ind_root         : {args.ind_root}")
    print(f"  pop_root         : {args.pop_root}")
    print(f"  cluster_root     : {args.cluster_root}")
    print(f"  cluster_scales   : {args.cluster_scales_root}")
    print(f"  cluster_map      : {args.cluster_map}")
    print(f"  eval_seed        : {args.eval_seed}")
    print(f"  save_dir         : {args.save_dir}")
    print(f"  patients         : {len(pids)}")
    print(f"{'=' * 72}\n")

    all_results = {}
    for pid in pids:
        res = _worker(
            pid,
            algo,
            args.pinn_dir,
            args.ind_root,
            args.pop_root,
            args.glb_scales,
            args.cluster_root,
            args.cluster_scales_root,
            cluster_map,
            args.n_eval,
            args.K,
            init_noise_std=args.init_noise_std,
            eval_seed=args.eval_seed,
        )
        if res is None:
            continue
        all_results[pid] = res
        _plot_per_patient(pid, res, per_patient_dir, algo)
        _save_trajectories(pid, res, per_patient_dir, algo)

    if not all_results:
        raise SystemExit("No results collected.")
    _plot_summary(all_results, summary_dir, algo, args.K, cluster_map=cluster_map)
    print(f"\nDone. Plots and trajectories saved under {args.save_dir}")


if __name__ == "__main__":
    main()
