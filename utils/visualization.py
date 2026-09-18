"""
utils/visualization.py — Visualization utilities

Includes:
  1. plot_pinn_fit   — single-step prediction + short rollout
  2. plot_pinn_fit2  — full rollout
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from data.config import state_features, state_dim, action_dim, action_features, feature_index


# ─────────────────────────────────────────────────────────────
# PINN fitting visualization
# ─────────────────────────────────────────────────────────────
def plot_pinn_fit(model, patient, scales, save_path=None):
    model.eval()
    data = patient["data"].clone()
    dt   = patient["dt"]
    x_full  = data[:, 1:]
    x       = x_full[:, :state_dim]
    a       = x_full[:, state_dim:]
    mask_x  = patient["mask"][:, 1:1+state_dim]
    t_raw   = data[:, 0].cpu().numpy()
    mean_np = scales["state_mean"].cpu().numpy()
    std_np  = scales["state_std"].cpu().numpy()

    with torch.no_grad():
        x_t         = x[:-1]
        a_t         = a[:-1]
        dt_t        = dt.unsqueeze(1)
        dxdt_p      = model(x_t, a_t)
        x_next_pred = x_t + dxdt_p * dt_t

    x_true_np = x.cpu().numpy()            * std_np + mean_np
    x_pred_np = x_next_pred.cpu().numpy()  * std_np + mean_np
    x_next_np = x[1:].cpu().numpy()        * std_np + mean_np
    mask_np   = mask_x.cpu().numpy()
    t_np      = t_raw[:-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (i, fname) in zip(axes.flatten(), enumerate(state_features)):
        obs_t    = mask_np[:-1, i] > 0.5
        obs_next = mask_np[1:,  i] > 0.5
        both     = obs_t & obs_next
        ax.scatter(t_raw[mask_np[:, i] > 0.5],
                   x_true_np[mask_np[:, i] > 0.5, i],
                   s=20, label="True", zorder=3, alpha=0.7)
        ax.scatter(t_np[both], x_pred_np[both, i],
                   s=10, marker='x', label="1-step pred",
                   color='tab:orange', zorder=2, alpha=0.7)
        if both.sum() > 0:
            err  = np.abs(x_pred_np[both, i] - x_next_np[both, i])
            rmse = np.sqrt((err**2).mean())
            ax.set_title(f"{fname}  1-step RMSE={rmse:.2f}")
        else:
            ax.set_title(fname)
        ax.set_xlabel("hours (norm)")
        ax.legend(fontsize=7)
    plt.suptitle(f"ICU {patient['icu_id']} — Single-step Predictions", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pinn_fit2(model, patient, scales, save_path=None):
    model.eval()
    data   = patient["data"].clone()
    dt     = patient["dt"]
    x_full = data[:, 1:]
    x      = x_full[:, :state_dim]
    a      = x_full[:, state_dim:]
    t_raw  = data[:, 0].cpu().numpy()
    mean_np = scales["state_mean"].cpu().numpy()
    std_np  = scales["state_std"].cpu().numpy()

    with torch.no_grad():
        preds = [x[0:1]]
        for i in range(len(x)-1):
            x_next = model.step(
                preds[-1], a[i:i+1], dt[i],
                scales["state_min"], scales["state_max"])
            preds.append(x_next)
        pred = torch.cat(preds, dim=0)

    x_orig    = x.cpu().numpy()    * std_np + mean_np
    pred_orig = pred.cpu().numpy() * std_np + mean_np
    mask_np   = patient["mask"].cpu().numpy()[:, 1:1+state_dim]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (i, fname) in zip(axes.flatten(), enumerate(state_features)):
        valid = mask_np[:, i] > 0.5
        ax.scatter(t_raw[valid], x_orig[valid, i], s=15, label="True", zorder=3)
        ax.plot(t_raw, pred_orig[:, i], lw=1.5, label="PINN rollout")
        ax.set_title(fname)
        ax.set_xlabel("hours (norm)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def visualize_episode1(agent, env, patient, save_dir="results", tag="sac"):
    """Run one deterministic episode and save state / action / dt / reward+SOFA plots."""
    from data.config import state_dim

    os.makedirs(save_dir, exist_ok=True)

    if patient is not None:
        state = env.reset(patient=patient)
    else:
        state = env.reset(init_state_norm=np.zeros(state_dim, dtype=np.float32))

    traj = {k: [] for k in [
        "t_start", "t_end", "dt",
        "spo2", "pao2", "bili", "gcs", "urine", "lac",
        "sofa", "reward",
        "u_fio2", "u_vaso", "u_flu",
    ]}

    while True:
        t0     = env.current_t
        option = agent.select_action(state, deterministic=True)
        state, r, done, info = env.step(option)
        t1 = env.current_t

        s = env.phys_state * env.std[:state_dim] + env.mean[:state_dim]
        u = info["u_norm"]

        traj["t_start"].append(t0);  traj["t_end"].append(t1)
        traj["dt"].append(info["dt"])
        traj["spo2"].append(float(s[feature_index["spo2"]]))
        traj["pao2"].append(float(s[feature_index["pao2"]]))
        traj["bili"].append(float(s[feature_index["bilirubin"]]))
        traj["gcs"].append(float(s[feature_index["gcs"]]))
        traj["urine"].append(float(s[feature_index["urine"]]))
        traj["lac"].append(float(s[feature_index["lactate"]]))
        traj["sofa"].append(float(info.get("sofa", float("nan"))))
        traj["reward"].append(float(r))
        traj["u_fio2"].append(float(u[0]))
        traj["u_vaso"].append(float(u[1]))
        traj["u_flu"].append(float(u[2]))

        if done:
            break

    traj = {k: np.array(v, dtype=np.float32) for k, v in traj.items()}
    t0s, t1s = traj["t_start"], traj["t_end"]
    n = len(t0s)

    # ── shared style ────────────────────────────────────────────────────────
    PLT_W, STEP_C = 11, "#2c7bb6"

    def _step(ax, ts, te, vals, color=STEP_C, lw=2.0, fill=True, alpha=0.15):
        """Zero-order hold (step) line with optional fill."""
        for i in range(len(ts)):
            ax.plot([ts[i], te[i]], [vals[i], vals[i]],
                    color=color, lw=lw, solid_capstyle="butt")
            if i < len(ts) - 1:
                ax.plot([te[i], te[i]], [vals[i], vals[i+1]],
                        color=color, lw=lw * 0.6, ls="--", alpha=0.5)
        if fill:
            for i in range(len(ts)):
                ax.fill_between([ts[i], te[i]], 0, vals[i],
                                color=color, alpha=alpha)

    def _ivlines(ax, ts):
        """Light grey intervention markers."""
        for t in ts[1:]:
            ax.axvline(t, color="#aaaaaa", lw=0.7, ls="--", zorder=0)

    def _finish(ax, xlabel="Time (hours)"):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.set_xlim(0, traj["t_end"][-1] * 1.02)

    # =========================================================
    # Figure 1: STATE  (3 × 2 grid)
    # =========================================================
    panels = [
        ("SpO₂ (%)",          "spo2",  {"thresholds": [(90, "#e74c3c", "SpO₂=90")]}),
        ("PaO₂ (mmHg)",       "pao2",  {}),
        ("Bilirubin (mg/dL)", "bili",  {"thresholds": [(1.2, "#e67e22", "Bili=1.2")]}),
        ("GCS",               "gcs",   {"thresholds": [(8, "#e74c3c", "GCS=8")],
                                         "ylim": (0, 16)}),
        ("Urine (mL/h)",      "urine", {}),
        ("Lactate (mmol/L)",  "lac",   {"thresholds": [(4.0, "#e67e22", "Lac=4"),
                                                        (8.0, "#e74c3c", "Lac=8")]}),
    ]

    fig1, axes1 = plt.subplots(3, 2, figsize=(PLT_W, 10), constrained_layout=True)
    fig1.suptitle(f"State trajectory  [{tag}]", fontsize=13, fontweight="bold", y=1.01)

    colors = ["#2c7bb6", "#4dac26", "#d01c8b", "#f1b6da", "#b8e186", "#fdae61"]
    for ax, (label, key, opts), col in zip(axes1.flatten(), panels, colors):
        _step(ax, t0s, t1s, traj[key], color=col)
        _ivlines(ax, t0s)
        for thr, tc, tlbl in opts.get("thresholds", []):
            ax.axhline(thr, color=tc, lw=1.0, ls=":", alpha=0.8,
                       label=tlbl)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        if "ylim" in opts:
            ax.set_ylim(*opts["ylim"])
        ax.set_ylabel(label, fontsize=9)
        _finish(ax)

    path1 = f"{save_dir}/state_{tag}.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Saved] {path1}")

    # =========================================================
    # Figure 2: ACTION  (3 stacked subplots)
    # =========================================================
    act_panels = [
        ("FiO₂ (norm)",      "u_fio2", "#e74c3c"),
        ("Vasopressor (norm)", "u_vaso", "#8e44ad"),
        ("Fluids (norm)",    "u_flu",  "#27ae60"),
    ]
    fig2, axes2 = plt.subplots(3, 1, figsize=(PLT_W, 6),
                                sharex=True, constrained_layout=True)
    fig2.suptitle(f"Action profiles  [{tag}]", fontsize=13, fontweight="bold")

    for ax, (label, key, col) in zip(axes2, act_panels):
        _step(ax, t0s, t1s, traj[key], color=col, fill=True, alpha=0.18)
        _ivlines(ax, t0s)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        _finish(ax)

    axes2[-1].set_xlabel("Time (hours)", fontsize=9)
    path2 = f"{save_dir}/action_{tag}.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Saved] {path2}")

    # =========================================================
    # Figure 3: δt  (bar + cumulative line)
    # =========================================================
    fig3, axes3 = plt.subplots(2, 1, figsize=(PLT_W, 5),
                                sharex=True, constrained_layout=True)
    fig3.suptitle(f"Intervention timing  [{tag}]", fontsize=13, fontweight="bold")

    ax_bar, ax_cum = axes3
    bar_w = np.minimum(traj["dt"] * 0.85, 1.5)
    ax_bar.bar(t0s, traj["dt"], width=bar_w, align="edge",
               color=STEP_C, alpha=0.75, edgecolor="white", lw=0.5)
    ax_bar.set_ylabel(r"$\Delta t$ (hours)", fontsize=9)
    _finish(ax_bar)

    cum_t = np.cumsum(traj["dt"])
    ax_cum.plot(t0s, cum_t, color="#e74c3c", lw=2.0, marker="o",
                markersize=3)
    ax_cum.set_ylabel("Cumul. time (h)", fontsize=9)
    ax_cum.set_xlabel("Intervention start (h)", fontsize=9)
    _finish(ax_cum)

    path3 = f"{save_dir}/dt_{tag}.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[Saved] {path3}")

    # =========================================================
    # Figure 4: SOFA + Cumulative Reward  (new)
    # =========================================================
    fig4, (ax_sofa, ax_rew) = plt.subplots(2, 1, figsize=(PLT_W, 5),
                                            sharex=True, constrained_layout=True)
    fig4.suptitle(f"SOFA & Reward  [{tag}]", fontsize=13, fontweight="bold")

    _step(ax_sofa, t0s, t1s, traj["sofa"], color="#c0392b", fill=True, alpha=0.15)
    _ivlines(ax_sofa, t0s)
    ax_sofa.set_ylabel("SOFA score", fontsize=9)
    _finish(ax_sofa)

    cum_rew  = np.cumsum(traj["reward"])
    t_rew    = np.concatenate([[0.0], t1s])       # prepend origin (0, 0)
    cum_rew_ = np.concatenate([[0.0], cum_rew])
    ax_rew.plot(t_rew, cum_rew_, color="#2980b9", lw=2.0, marker="o", markersize=3)
    ax_rew.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax_rew.fill_between(t_rew, cum_rew_, 0,
                         where=cum_rew_ < 0, color="#e74c3c", alpha=0.12)
    ax_rew.fill_between(t_rew, cum_rew_, 0,
                         where=cum_rew_ >= 0, color="#27ae60", alpha=0.12)
    ax_rew.set_ylabel("Cumulative reward", fontsize=9)
    ax_rew.set_xlabel("Time (hours)", fontsize=9)
    _finish(ax_rew)

    path4 = f"{save_dir}/sofa_reward_{tag}.png"
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"[Saved] {path4}")

    # ── save as npz (full) ────────────────────────────────────────────────
    npz_path = f"{save_dir}/traj_{tag}.npz"
    np.savez(npz_path, **traj)
    print(f"[Saved] {npz_path}")

    # ── save as csv ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "t_start":  traj["t_start"],
        "t_end":    traj["t_end"],
        "dt":       traj["dt"],
        "spo2":     traj["spo2"],
        "pao2":     traj["pao2"],
        "bili":     traj["bili"],
        "gcs":      traj["gcs"],
        "urine":    traj["urine"],
        "lactate":  traj["lac"],
        "sofa":     traj["sofa"],
        "reward":   traj["reward"],
        "fio2":     traj["u_fio2"],
        "vaso":     traj["u_vaso"],
        "fluids":   traj["u_flu"],
    })
    csv_path = f"{save_dir}/traj_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    return traj


def plot_loss_history(loss_history, save_path) -> None:
    """Plot and save PINN training loss curve.

    Accepts either a list of total losses (legacy) or a dict of lists
    with keys ``{"total", "data", "roll", "ode", "smooth"}``.
    """
    if isinstance(loss_history, dict):
        total = loss_history.get("total", [])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(total, lw=1)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("PINN training loss (total)")
        axes[0].set_yscale("log")
        axes[0].grid(alpha=0.3)
        for key, color in [("data", "steelblue"), ("roll", "darkorange"),
                           ("ode", "green"), ("smooth", "gray")]:
            ys = loss_history.get(key, [])
            if ys:
                axes[1].plot(ys, "-", color=color, lw=1.2, alpha=0.85, label=key)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Epoch"); axes[1].set_title("Loss components")
        axes[1].grid(alpha=0.3); axes[1].legend()
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        print(f"  [Saved] {save_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN training loss")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  [Saved] {save_path}")