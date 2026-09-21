#!/usr/bin/env python3
"""Compare Global PINN vs Per-Patient PINN — combined plots.

Generates two figures:
  1. PINN Fit: True observations vs Global PINN vs Per-Patient PINN (2x3 grid)
  2. RL Evaluation: Global vs Per-Patient, vardt vs fixdt on one figure (4 columns)

Usage
-----
    python scripts/plot_global_vs_perpatient.py --icu_id 200065
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch

from data.config import state_dim, action_dim, device, state_features, feature_index


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: PINN Fit — Global vs Per-Patient
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pinn_fit_combined(
    csv_path: str,
    icu_id: int,
    global_pinn_path: Path | None,
    perpatient_pinn_path: Path,
    global_scales_path: Path | None,
    perpatient_scales_path: Path | None,
    save_path: str,
    cluster_pinn_path: Path | None = None,
    cluster_scales_path: Path | None = None,
    figsize: tuple[float, float] = (20.0, 9.0),
    title: str | None = None,
    annotation: str | None = None,
    axis_label_size: float | None = None,
    x_label_size: float = 9.0,
    y_label_size: float = 10.0,
    tick_label_size: float | None = None,
    legend_size: float = 8.0,
    title_size: float = 14.0,
    annotation_size: float = 14.0,
    color_true: str = "#4575b4",
    color_pop: str = "#e66101",
    color_clu: str = "#3182bd",
    color_ind: str = "#1b9e77",
    label_true: str = "True",
    label_pop: str = "Population PINN",
    label_clu: str = "Clu PINN",
    label_ind: str = "Individual PINN",
    legend_panel: bool = False,
    sofa_ylabel: str = "SOFA (smooth)",
    train_ratio: float = 1.0,
) -> None:
    """Each PINN gets data re-normalised with its own training scales.

    Note: Individual (per-patient) PINNs are trained with scales derived
    *dynamically from this patient's own CSV rows* (see ``data/loader.py``),
    not with the scales saved in ``cluster_scales.npy`` (which are typically
    cluster-level statistics).  Therefore, when ``perpatient_scales_path``
    is ``None``, scales are recomputed from the CSV — this is the correct
    behaviour for Individual PINNs.  Population PINNs should still pass
    ``global_scales_path`` (cluster-level npy) since those PINNs were
    trained with the saved cluster scales.
    """
    import pandas as pd
    from data.config import action_features, ZERO_IS_NAN
    from models.pinn import PINN

    df = pd.read_csv(csv_path)
    pdf = df[df["icu_id"] == icu_id].sort_values("hours")
    hours_real = pdf["hours"].values.astype(np.float32)
    test_start_idx = int(len(hours_real) * train_ratio) if train_ratio < 1.0 else 0

    sv_raw = pdf[state_features].values.astype(np.float32)
    for i, f in enumerate(state_features):
        if f in ZERO_IS_NAN:
            sv_raw[:, i] = np.where(sv_raw[:, i] == 0, np.nan, sv_raw[:, i])
    state_mask = (~np.isnan(sv_raw)).astype(np.float32)

    av_raw = pdf[action_features].values.astype(np.float32)
    av_raw = np.where(np.isnan(av_raw), 0.0, av_raw)
    av_raw = np.clip(av_raw, 0.0, None)

    dt_real = np.diff(hours_real).astype(np.float32)
    dt_real = np.clip(dt_real, 1e-3, 24.0)
    dt_t = torch.tensor(dt_real, dtype=torch.float32).to(device)

    def _scales_from_csv() -> dict:
        """Recompute scales the same way ``data/loader.py`` does."""
        mean = np.nanmean(sv_raw, axis=0).astype(np.float32)
        std = np.nanstd(sv_raw, axis=0).astype(np.float32)
        std[std == 0] = 1.0
        smin = np.nanmin(sv_raw, axis=0)
        smax = np.nanmax(sv_raw, axis=0)
        smin_norm = ((smin - mean) / std).astype(np.float32)
        smax_norm = ((smax - mean) / std).astype(np.float32)
        ascl = np.nanmax(np.log1p(av_raw), axis=0).astype(np.float32)
        ascl[ascl == 0] = 1.0
        return {
            "mean": mean, "std": std, "ascl": ascl,
            "state_min": smin_norm, "state_max": smax_norm,
        }

    def _load_scales(path: Path | None) -> dict:
        if path is None:
            return _scales_from_csv()
        if path.suffix == ".npy":
            sc = np.load(path, allow_pickle=True).item()
            return {
                "mean": sc["mean"].astype(np.float32),
                "std": sc["std"].astype(np.float32),
                "ascl": sc["ascl"].astype(np.float32),
                "state_min": sc.get("state_min", np.full(state_dim, -5, dtype=np.float32)).astype(np.float32),
                "state_max": sc.get("state_max", np.full(state_dim, 5, dtype=np.float32)).astype(np.float32),
            }
        else:
            raw = np.load(path)
            return {
                "mean": raw["state_mean"].astype(np.float32),
                "std": raw["state_std"].astype(np.float32),
                "ascl": raw["action_scale"].astype(np.float32),
                "state_min": raw["state_min"].astype(np.float32),
                "state_max": raw["state_max"].astype(np.float32),
            }

    def _rollout(pinn_path, scales):
        pinn = PINN(state_dim, action_dim).to(device)
        pinn.load_state_dict(torch.load(pinn_path, map_location=device, weights_only=False))
        pinn.eval()

        sv_imp = np.where(np.isnan(sv_raw), scales["mean"], sv_raw)
        x_norm = (sv_imp - scales["mean"]) / scales["std"]
        a_norm = np.log1p(av_raw) / scales["ascl"]

        x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)
        a_t = torch.tensor(a_norm, dtype=torch.float32).to(device)
        s_min = torch.tensor(scales["state_min"], dtype=torch.float32).to(device)
        s_max = torch.tensor(scales["state_max"], dtype=torch.float32).to(device)

        with torch.no_grad():
            preds = [x_t[0:1]]
            for i in range(len(x_t) - 1):
                x_next = pinn.step(preds[-1], a_t[i:i + 1], dt_t[i], s_min, s_max)
                preds.append(x_next)
            pred = torch.cat(preds, dim=0)

        return pred.cpu().numpy() * scales["std"] + scales["mean"]

    def _compute_sofa(phys_states: np.ndarray, vaso_raw: np.ndarray) -> np.ndarray:
        """Compute smooth SOFA score from physical-scale values (same logic as ICUEnvironment)."""
        import math

        def sig(x):
            return 1.0 / (1.0 + math.exp(-float(np.clip(x, -50, 50))))

        def smooth_step(x, thr, ws, wl, alpha=0.7):
            return alpha * sig((x - thr) / ws) + (1 - alpha) * sig((x - thr) / wl)

        fi = {f: idx for idx, f in enumerate(state_features)}
        T = len(phys_states)
        sofa = np.zeros(T, dtype=np.float32)
        for t in range(T):
            spo2  = float(phys_states[t, fi["SpO2"]])
            bili  = float(phys_states[t, fi["Bilirubin"]])
            gcs   = float(phys_states[t, fi["GCS"]])
            urine = float(phys_states[t, fi["Urine_Step"]])
            vaso  = float(vaso_raw[t])

            sr = (smooth_step(94 - spo2, 0, 1.0, 3.0) +
                  smooth_step(90 - spo2, 0, 1.0, 3.0) +
                  smooth_step(85 - spo2, 0, 1.0, 3.0) +
                  smooth_step(80 - spo2, 0, 1.0, 3.0)) * 0.7
            sl = (smooth_step(bili,  1.2, 0.3, 1.5) +
                  smooth_step(bili,  2.0, 0.3, 1.5) +
                  smooth_step(bili,  6.0, 0.3, 1.5) +
                  smooth_step(bili, 12.0, 0.3, 1.5))
            sn = (smooth_step(15 - gcs, 0, 0.5, 1.5) +
                  smooth_step(13 - gcs, 0, 0.5, 1.5) +
                  smooth_step(10 - gcs, 0, 0.5, 1.5) +
                  smooth_step( 6 - gcs, 0, 0.5, 1.5))
            sk = (smooth_step(500 - urine, 0, 50.0, 120.0) +
                  2 * smooth_step(200 - urine, 0, 50.0, 120.0))
            sc = (smooth_step(vaso, 0.0,  0.02, 0.1) * 2 +
                  smooth_step(vaso, 0.1,  0.03, 0.15) +
                  smooth_step(vaso, 0.25, 0.05, 0.2))
            sofa[t] = sr + sl + sn + sk + sc
        return sofa

    sc_pp = _load_scales(perpatient_scales_path)

    sv_true = np.where(np.isnan(sv_raw), np.nan, sv_raw)

    pred_global = None
    if global_pinn_path is not None and global_scales_path is not None:
        sc_global = _load_scales(global_scales_path)
        pred_global = _rollout(global_pinn_path, sc_global)
    pred_perpatient = _rollout(perpatient_pinn_path, sc_pp)
    pred_cluster = None
    if cluster_pinn_path is not None and cluster_scales_path is not None:
        sc_cluster = _load_scales(cluster_scales_path)
        pred_cluster = _rollout(cluster_pinn_path, sc_cluster)

    vaso_raw = av_raw[:, 1]
    sv_true_imp = np.where(np.isnan(sv_raw), 0.0, sv_raw)
    sofa_true = _compute_sofa(sv_true_imp, vaso_raw)
    sofa_global = _compute_sofa(pred_global, vaso_raw) if pred_global is not None else None
    sofa_pp = _compute_sofa(pred_perpatient, vaso_raw)
    sofa_cluster = _compute_sofa(pred_cluster, vaso_raw) if pred_cluster is not None else None

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    if title is None:
        if pred_global is None and pred_cluster is not None:
            title = f"PINN Fit: Clu vs Individual — ICU stay {icu_id}"
        elif pred_cluster is not None:
            title = f"PINN Fit: Population vs Clu vs Individual — ICU stay {icu_id}"
        else:
            title = f"PINN Fit: Population vs Individual — ICU stay {icu_id}"
    if title:
        fig.suptitle(
            title,
            fontsize=title_size, fontweight="bold",
        )
    if annotation:
        fig.text(
            0.5,
            0.925,
            annotation,
            ha="center",
            va="top",
            fontsize=annotation_size,
        )
    label_size_x = axis_label_size if axis_label_size is not None else x_label_size
    label_size_y = axis_label_size if axis_label_size is not None else y_label_size

    for ax, (i, fname) in zip(axes.flatten(), enumerate(state_features)):
        valid = state_mask[:, i] > 0.5

        ax.scatter(hours_real[valid], sv_true[valid, i],
                   s=30, marker="*", color=color_true, alpha=0.7,
                   label=label_true, zorder=3)
        if pred_global is not None:
            ax.plot(hours_real, pred_global[:, i],
                    color=color_pop, lw=1.8, alpha=0.85, label=label_pop)
        if pred_cluster is not None:
            ax.plot(hours_real, pred_cluster[:, i],
                    color=color_clu, lw=1.8, ls="-.", alpha=0.85,
                    label=label_clu)
        ax.plot(hours_real, pred_perpatient[:, i],
                color=color_ind, lw=1.8, ls="--", alpha=0.85,
                label=label_ind)
        if test_start_idx > 0:
            ax.axvspan(hours_real[test_start_idx], hours_real[-1],
                       alpha=0.08, color="lightblue", zorder=0)
            ax.axvline(x=hours_real[test_start_idx], color="gray", ls="--", lw=1.5, alpha=0.7)

        ax.set_ylabel(fname, fontsize=label_size_y)
        ax.set_xlabel("Time (hours)", fontsize=label_size_x)
        if tick_label_size is not None:
            ax.tick_params(axis="both", labelsize=tick_label_size)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    ax_sofa = axes[1, 2]
    ax_sofa.scatter(hours_real, sofa_true,
                    s=30, marker="*", color=color_true, alpha=0.7,
                    label=label_true, zorder=3)
    if sofa_global is not None:
        ax_sofa.plot(hours_real, sofa_global,
                     color=color_pop, lw=1.8, alpha=0.85, label=label_pop)
    if sofa_cluster is not None:
        ax_sofa.plot(hours_real, sofa_cluster,
                     color=color_clu, lw=1.8, ls="-.", alpha=0.85,
                     label=label_clu)
    ax_sofa.plot(hours_real, sofa_pp,
                 color=color_ind, lw=1.8, ls="--", alpha=0.85,
                 label=label_ind)
    if test_start_idx > 0:
        ax_sofa.axvspan(hours_real[test_start_idx], hours_real[-1],
                        alpha=0.08, color="lightblue", zorder=0)
        ax_sofa.axvline(x=hours_real[test_start_idx], color="gray", ls="--", lw=1.5, alpha=0.7)
    ax_sofa.set_ylabel(sofa_ylabel, fontsize=label_size_y)
    ax_sofa.set_xlabel("Time (hours)", fontsize=label_size_x)
    if tick_label_size is not None:
        ax_sofa.tick_params(axis="both", labelsize=tick_label_size)
    ax_sofa.grid(alpha=0.2)
    ax_sofa.spines[["top", "right"]].set_visible(False)

    if legend_panel:
        legend_ax = axes[1, 3]
        handles, labels = axes[0, 0].get_legend_handles_labels()
        legend_ax.axis("off")
        legend_ax.legend(
            handles,
            labels,
            fontsize=legend_size,
            loc="center",
            framealpha=0.8,
            edgecolor="none",
        )
    else:
        axes[1, 3].set_visible(False)
        ax_sofa.legend(fontsize=legend_size, loc="best", framealpha=0.8, edgecolor="none")
        axes[0, 0].legend(fontsize=legend_size, loc="best", framealpha=0.8, edgecolor="none")

    top = 0.94 if title else 1.0
    if annotation:
        top = min(top, 0.9)
    plt.tight_layout(rect=[0, 0, 1, top])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: RL Evaluation — Global vs Per-Patient, vardt vs fixdt
# ═══════════════════════════════════════════════════════════════════════════════

def _load_agg(path: Path) -> dict | None:
    if not path.exists():
        print(f"[WARN] Not found: {path}")
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def plot_rl_comparison_combined(
    icu_id: int,
    global_root: str,
    perpatient_root: str,
    K: int,
    save_path: str,
    lactate_threshold: float = 4.0,
    K_pop: int | None = None,
    K_ind: int | None = None,
    file_prefix: str = "lagrangian_trpo",
    include_cluster: bool = False,
    include_population: bool = True,
    npz_paths: dict | None = None,
) -> None:
    """Single fig9-style layout with overlaid policy rollouts.

    ``file_prefix`` selects which algorithm's aggregated eval files to use,
    e.g. ``"lagrangian_trpo"``, ``"sac"``, ``"trpo"``, ``"ppo"``,
    ``"lagrangian_ppo"``.

    If ``K_pop`` / ``K_ind`` are set, they override ``K`` for population /
    individual aggregated eval paths respectively (e.g. K_pop=12, K_ind=20).

    ``npz_paths`` bypasses all path-construction logic. Pass a dict mapping
    label → Path for each series to plot, e.g.::

        npz_paths={
            "ind_vardt": Path(...), "ind_fixdt": Path(...),
            "pop_vardt": Path(...), "pop_fixdt": Path(...),
            "clu_vardt": Path(...), "clu_fixdt": Path(...),
        }
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker

    if npz_paths is not None:
        data = {}
        for label, path in npz_paths.items():
            d = _load_agg(Path(path))
            if d is None:
                return
            data[label] = d
    else:
        k_pop = K if K_pop is None else K_pop
        k_ind = K if K_ind is None else K_ind

        cross_eval_dir = Path(perpatient_root) / f"patient_{icu_id}"
        pop_vardt_cross = cross_eval_dir / f"{file_prefix}_vardt_K{k_ind}" / "cross_eval_pop" / "aggregated.npz"
        pop_fixdt_cross = cross_eval_dir / f"{file_prefix}_fixdt_K{k_ind}" / "cross_eval_pop" / "aggregated.npz"
        use_cross = include_population and pop_vardt_cross.exists() and pop_fixdt_cross.exists()
        clu_vardt_cross = cross_eval_dir / f"{file_prefix}_vardt_K{k_ind}" / "cross_eval_clu" / "aggregated.npz"
        clu_fixdt_cross = cross_eval_dir / f"{file_prefix}_fixdt_K{k_ind}" / "cross_eval_clu" / "aggregated.npz"
        use_clu_cross = include_cluster and clu_vardt_cross.exists() and clu_fixdt_cross.exists()

        if use_cross or (not include_population and use_clu_cross):
            msg = "  [Using cross-eval: Pop policy on Ind PINN"
            if not include_population:
                msg = "  [Using cross-eval: Clu policy on Ind PINN"
            elif use_clu_cross:
                msg += " + Clu policy"
            msg += "]"
            print(msg)
            configs = []
            if include_population:
                configs.extend([
                    ("pop_vardt",  pop_vardt_cross),
                    ("pop_fixdt",  pop_fixdt_cross),
                ])
            if use_clu_cross:
                configs.extend([
                    ("clu_vardt", clu_vardt_cross),
                    ("clu_fixdt", clu_fixdt_cross),
                ])
            configs.extend([
                ("ind_vardt",  Path(perpatient_root) / f"patient_{icu_id}" / f"{file_prefix}_vardt_K{k_ind}" / "eval" / "aggregated.npz"),
                ("ind_fixdt",  Path(perpatient_root) / f"patient_{icu_id}" / f"{file_prefix}_fixdt_K{k_ind}" / "eval" / "aggregated.npz"),
            ])
        else:
            configs = []
            if include_population:
                configs.extend([
                    ("pop_vardt",  Path(global_root) / f"{file_prefix}_vardt_K{k_pop}" / "eval" / "aggregated.npz"),
                    ("pop_fixdt",  Path(global_root) / f"{file_prefix}_fixdt_K{k_pop}" / "eval" / "aggregated.npz"),
                ])
            configs.extend([
                ("ind_vardt",  Path(perpatient_root) / f"patient_{icu_id}" / f"{file_prefix}_vardt_K{k_ind}" / "eval" / "aggregated.npz"),
                ("ind_fixdt",  Path(perpatient_root) / f"patient_{icu_id}" / f"{file_prefix}_fixdt_K{k_ind}" / "eval" / "aggregated.npz"),
            ])

        data = {}
        for label, path in configs:
            d = _load_agg(path)
            if d is None:
                return
            data[label] = d

    C_POP_V = "#1f5aa6"
    C_POP_F = "#8fb8e8"
    C_IND_V = "#c81d25"
    C_IND_F = "#f28e8c"
    C_CLU_V = "#1f7a3a"
    C_CLU_F = "#8fd19e"
    ALPHA = 0.12
    AXIS_LABEL_SIZE = 16   # scaled ×0.90 with figsize height 10→9
    TICK_LABEL_SIZE = 14
    LEGEND_SIZE = 11
    TITLE_SIZE = 16

    line_cfgs = []
    if "pop_vardt" in data and "pop_fixdt" in data:
        line_cfgs.extend([
            ("pop_vardt",  C_POP_V, "-",  2.2),
            ("pop_fixdt", C_POP_F, "-",  2.2),
        ])
    if "clu_vardt" in data and "clu_fixdt" in data:
        line_cfgs.extend([
            ("clu_vardt", C_CLU_V, "-.", 2.1),
            ("clu_fixdt", C_CLU_F, "-.", 2.1),
        ])
    line_cfgs.extend([
        ("ind_vardt",  C_IND_V, "--", 2.0),
        ("ind_fixdt", C_IND_F, "--", 2.0),
    ])

    state_keys   = ["spo2", "pao2", "bilirubin", "gcs", "urine", "lactate"]
    action_keys  = ["fio2", "vaso", "fluids"]
    state_labels = ["SpO2", "PaO2", "Bilirubin", "GCS", "Urine", "Lactate"]
    action_labels = ["FiO2", "Vaso", "Fluids"]

    # 4-row × 3-col grid:
    #   rows 0-1 → states (2×3 = 6 subplots)
    #   row  2   → actions (1×3 = 3 subplots)
    #   row  3   → sofa (col 0) + lactate (col 1) + legend (col 2)
    # Natural cell aspect ~2.1:1 from figsize=(14,10) + spacing below.
    # wspace=0.20 adds ~1 character gap between each column pair.
    # Titles use x=-0.13 (axes coords) to shift ~4 chars left of axes edge.
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.35, wspace=0.20,
        top=0.93, bottom=0.06, left=0.07, right=1.0,
    )

    named_axes: list[tuple[str, object]] = []

    def _plot_quad(
        ax,
        key,
        ylabel,
        threshold=None,
        anchor=None,
        labelpad=6,
    ):
        means = []
        for tag, color, ls, lw in line_cfgs:
            t_grid = data[tag]["t_grid"]
            m = data[tag][f"{key}_mean"]
            s = data[tag][f"{key}_std"]
            means.append(m)
            ax.plot(t_grid, m, color=color, ls=ls, lw=lw)
            if np.max(s) > 1e-4 * (np.max(np.abs(m)) + 1e-8):
                ax.fill_between(t_grid, m - s, m + s, color=color, alpha=ALPHA)
        if threshold is not None:
            ax.axhline(y=threshold, color="black", ls="--", lw=1.4, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=labelpad)
        ax.grid(alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=TICK_LABEL_SIZE, pad=2)
        if anchor is not None:
            ax.set_anchor(anchor)
        try:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        except AttributeError:
            pass
        if key == "bilirubin" and means:
            y = np.concatenate([np.asarray(m).ravel() for m in means])
            center = round(float(np.nanmean(y)), 2)
            spread = float(np.nanmax(np.abs(y - center)))
            margin = max(spread * 1.15, 0.005)
            ax.set_ylim(center - margin, center + margin)
            ax.set_yticks([center])
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    _TITLE_X = -0.13  # ~4 character widths left of the axes left edge

    # Rows 0-1: state plots (spo2, pao2, bilirubin, gcs, urine, lactate)
    for si, (key, label) in enumerate(zip(state_keys, state_labels)):
        r, c = divmod(si, 3)
        ax = fig.add_subplot(gs[r, c])
        _plot_quad(ax, key, label)
        named_axes.append((f"state_{key}", ax))
        if si == 0:
            ax.set_title("(a) State", fontsize=TITLE_SIZE, fontweight="bold",
                         x=_TITLE_X, ha="left", pad=1)
        if r == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time(hours)", fontsize=AXIS_LABEL_SIZE, labelpad=4)

    # Row 2: action plots (fio2, vaso, fluids)
    for ai, (key, label) in enumerate(zip(action_keys, action_labels)):
        ax = fig.add_subplot(gs[2, ai])
        _plot_quad(ax, key, label)
        named_axes.append((f"action_{key}", ax))
        if ai == 0:
            ax.set_title("(b) Action", fontsize=TITLE_SIZE, fontweight="bold",
                         x=_TITLE_X, ha="left", pad=1)
        ax.set_xlabel("Time(hours)", fontsize=AXIS_LABEL_SIZE, labelpad=4)

    # Row 3: sofa (col 0), lactate/safety (col 1), legend (col 2)
    ax_sofa = fig.add_subplot(gs[3, 0])
    _plot_quad(ax_sofa, "sofa", "SOFA")
    ax_sofa.set_title("(c) SOFA", fontsize=TITLE_SIZE, fontweight="bold",
                      x=_TITLE_X, ha="left", pad=1)
    ax_sofa.set_xlabel("Time(hours)", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    named_axes.append(("outcome_sofa", ax_sofa))

    ax_lac = fig.add_subplot(gs[3, 1])
    _plot_quad(ax_lac, "lactate", "Lactate\n(safety)", threshold=lactate_threshold)
    ax_lac.set_title("(d) Safety", fontsize=TITLE_SIZE, fontweight="bold",
                     x=_TITLE_X, ha="left", pad=1)
    ax_lac.set_xlabel("Time(hours)", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    named_axes.append(("outcome_lactate", ax_lac))

    handles = [
        mlines.Line2D([], [], color=C_IND_F, ls="--", lw=2, label="Ind-F"),
        mlines.Line2D([], [], color=C_IND_V, ls="--", lw=2, label="Ind-A"),
    ]
    if "clu_vardt" in data and "clu_fixdt" in data:
        handles.extend([
            mlines.Line2D([], [], color=C_CLU_F, ls="-.", lw=2, label="Clu-F"),
            mlines.Line2D([], [], color=C_CLU_V, ls="-.", lw=2, label="Clu-A"),
        ])
    if "pop_vardt" in data and "pop_fixdt" in data:
        handles.extend([
            mlines.Line2D([], [], color=C_POP_F, ls="-",  lw=2, label="Pop-F"),
            mlines.Line2D([], [], color=C_POP_V, ls="-",  lw=2, label="Pop-A"),
        ])
    handles.append(mlines.Line2D([], [], color="black", lw=1.2, ls="--", label="Threshold"))
    ax_legend = fig.add_subplot(gs[3, 2])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=handles, loc="center", fontsize=LEGEND_SIZE,
        ncol=2, handlelength=2.2, framealpha=0.8,
        edgecolor="none", columnspacing=1.0,
    )

    # Save combined figure at full figure width (no right-side clipping).
    plt.savefig(save_path, dpi=200)
    print(f"[Saved] {save_path}")

    # ── Save individual subplots ──────────────────────────────────────────────
    sp = Path(save_path)
    subplot_dir = sp.parent / sp.stem
    subplot_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for name, ax in named_axes:
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        # Add a small right-side margin (6 display px) to avoid clipping.
        bbox_r = type(bbox).from_extents(bbox.x0, bbox.y0, bbox.x1 + 6, bbox.y1)
        bbox_inches = bbox_r.transformed(fig.dpi_scale_trans.inverted())
        sub_path = subplot_dir / f"{name}.png"
        fig.savefig(sub_path, dpi=200, bbox_inches=bbox_inches)
    print(f"[Saved subplots] {subplot_dir}/")

    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/yoshihiko/medrl-tacos-old/data/mimic_pinn_v4_filtered.csv")
    p.add_argument("--icu_id", type=int, default=200065)
    p.add_argument("--global_pinn_dir", default="results/results_allpatient_pinn_v2")
    p.add_argument("--global_rl_root", default="results/lagrangian_trpo_compare")
    p.add_argument("--perpatient_root", default="results/single_patient")
    p.add_argument(
        "--perpatient_pinn_dir",
        default=None,
        help="Directory with pinn.pt and cluster_scales.npy (or scales.npz); "
             "if set, overrides perpatient_root/patient_{icu_id}/pinn",
    )
    p.add_argument("--K", type=int, default=12, help="Fallback K when K_pop/K_ind omitted")
    p.add_argument("--K_pop", type=int, default=None, help="Population RL eval folder K (e.g. 12)")
    p.add_argument("--K_ind", type=int, default=None, help="Individual RL eval folder K (e.g. 20)")
    p.add_argument("--max_patients_load", type=int, default=500)
    p.add_argument("--save_dir", default="results/combined_comparison")
    p.add_argument("--skip_pinn_fit", action="store_true")
    p.add_argument("--skip_rl", action="store_true")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── PINN Fit Combined ─────────────────────────────────────
    if not args.skip_pinn_fit:
        print("[Phase 1] PINN fit comparison plot …")
        if args.perpatient_pinn_dir:
            perpatient_pinn_dir = Path(args.perpatient_pinn_dir)
            perpatient_scales = perpatient_pinn_dir / "cluster_scales.npy"
            if not perpatient_scales.exists():
                perpatient_scales = perpatient_pinn_dir / "scales.npz"
        else:
            perpatient_pinn_dir = Path(args.perpatient_root) / f"patient_{args.icu_id}" / "pinn"
            perpatient_scales = perpatient_pinn_dir / "scales.npz"

        plot_pinn_fit_combined(
            csv_path=args.csv,
            icu_id=args.icu_id,
            global_pinn_path=Path(args.global_pinn_dir) / "pinn.pt",
            perpatient_pinn_path=perpatient_pinn_dir / "pinn.pt",
            global_scales_path=Path(args.global_pinn_dir) / "cluster_scales.npy",
            perpatient_scales_path=perpatient_scales,
            save_path=str(save_dir / f"pinn_fit_global_vs_perpatient_pid{args.icu_id}.png"),
        )

    # ── RL Evaluation Combined ────────────────────────────────
    if not args.skip_rl:
        print("[Phase 2] RL evaluation comparison plot …")
        k_pop = args.K_pop if args.K_pop is not None else args.K
        k_ind = args.K_ind if args.K_ind is not None else args.K
        tag = f"Kpop{k_pop}_Kind{k_ind}" if (args.K_pop is not None or args.K_ind is not None) else f"K{args.K}"
        plot_rl_comparison_combined(
            icu_id=args.icu_id,
            global_root=args.global_rl_root,
            perpatient_root=args.perpatient_root,
            K=args.K,
            K_pop=args.K_pop,
            K_ind=args.K_ind,
            save_path=str(save_dir / f"rl_global_vs_perpatient_pid{args.icu_id}_{tag}.png"),
        )

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
