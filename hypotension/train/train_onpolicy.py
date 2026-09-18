"""On-policy training loop for the hypotension benchmark (LAG-TRPO / TRPO / LAG-PPO).

Key differences from the sepsis train_onpolicy:
  1. Cost from env.step()["cost"] (Lac/UO/PaO2/hypertension safety).
  2. Tracks per-episode MAP and saves a training-curve PNG alongside policy.pt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _sample_init(init_state_norm: np.ndarray) -> np.ndarray:
    """2-D (N, state_dim) -> random row; 1-D -> return as-is."""
    arr = np.asarray(init_state_norm, dtype=np.float32)
    return arr[np.random.randint(arr.shape[0])] if arr.ndim == 2 else arr


def _smooth(vals: list[float], w: int = 20) -> np.ndarray:
    """Trailing-window moving average."""
    arr = np.array(vals, dtype=np.float64)
    if len(arr) == 0:
        return arr
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = arr[max(0, i - w + 1): i + 1].mean()
    return out


def _save_training_plot(
    save_path: Path,
    agent_name: str,
    step_at_ep: list[int],
    ep_rewards: list[float],
    ep_maps:    list[float],
    ep_costs:   list[float],
    ep_lambdas: list[float],
    cost_limit: float,
    smooth_w:   int = 20,
) -> None:
    """Save a 4-panel training-curve figure to save_path/training_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array(step_at_ep)
    has_lambda = any(not np.isnan(v) for v in ep_lambdas)
    n_panels = 4 if has_lambda else 3

    fig, axes = plt.subplots(n_panels, 1,
                              figsize=(10, 3.0 * n_panels),
                              constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Training curves — {agent_name}", fontsize=12, fontweight="bold")

    def _plot(ax, raw, label, ylabel, color, hline=None, hline_label=None,
              hline_color="red"):
        sm = _smooth(raw, smooth_w)
        ax.plot(steps, raw, alpha=0.25, color=color, lw=0.8)
        ax.plot(steps, sm,  alpha=0.90, color=color, lw=1.8, label=f"{label} (smooth-{smooth_w})")
        if hline is not None:
            ax.axhline(hline, color=hline_color, lw=1.2, ls="--",
                       label=hline_label or str(hline))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(steps[0] if len(steps) else 0, steps[-1] if len(steps) else 1)

    _plot(axes[0], ep_rewards, "return", "Episode return", "#4e79a7")
    _plot(axes[1], ep_maps, "MAP", "Mean MAP (mmHg)", "#59a14f",
          hline=65.0, hline_label="target >= 65 mmHg", hline_color="#d62728")
    _plot(axes[2], ep_costs, "cost", "Mean cost / step", "#e15759",
          hline=cost_limit, hline_label=f"cost limit = {cost_limit}",
          hline_color="#ff7f0e")
    if has_lambda:
        _plot(axes[3], ep_lambdas, "lambda", "Lagrange multiplier lambda", "#9467bd",
              hline=0.0, hline_label="lambda = 0", hline_color="gray")

    axes[-1].set_xlabel("Training steps", fontsize=9)

    out = save_path / "training_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out}")


# ── main training loop ────────────────────────────────────────────────────────

def train_onpolicy_hypo(
    agent,
    env,
    init_state_norm: np.ndarray,
    total_steps:  int   = 200_000,
    rollout_len:  int   = 2_048,
    save_dir:     str   = "results",
    agent_name:   str   = "LAGRANGIAN_TRPO_fixdt",
) -> None:
    """Option-SMDP on-policy loop; saves policy.pt + training_curves.png.

    init_state_norm : (state_dim,) single patient  or  (N, state_dim) multi.
    """
    _CONSTRAINED = {"LAGRANGIAN_PPO", "LAGRANGIAN_TRPO"}
    is_constrained = any(agent_name.upper().startswith(n) for n in _CONSTRAINED)
    cost_limit = float(getattr(agent, "cost_limit", 0.1))

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    step_at_ep: list[int]   = []
    ep_rewards: list[float] = []
    ep_maps:    list[float] = []
    ep_costs:   list[float] = []
    ep_lambdas: list[float] = []

    step_count = 0

    print(f"\n{'='*60}")
    print(f"  [{agent_name}]  steps={total_steps}  rollout={rollout_len}")
    print(f"  smdp_dim={env.smdp_state_dim}  opt_dim={env.option_action_dim}"
          f"  dt=[{env.dt_min},{env.dt_max}]h  cost_limit={cost_limit}")
    print(f"{'='*60}")

    while step_count < total_steps:
        obs = env.reset(init_state_norm=_sample_init(init_state_norm))
        ep_r = 0.0
        ep_c_sum = 0.0
        ep_n    = 0
        ep_map_list: list[float] = []
        done = False

        while not done and step_count < total_steps:
            if is_constrained:
                action, log_prob, value, cost_value = agent.select_action(obs)
            else:
                action, log_prob, value = agent.select_action(obs)

            next_obs, reward, done, info = env.step(action)
            cost = info["cost"]

            if is_constrained:
                agent.store_transition(obs, action, reward, cost,
                                       next_obs, float(done), log_prob, value, cost_value)
                ep_c_sum += cost
            else:
                agent.store_transition(obs, action, reward,
                                       next_obs, float(done), log_prob, value)

            obs         = next_obs
            ep_r       += reward
            ep_n       += 1
            ep_map_list.append(info.get("map", 0.0))
            step_count += 1

            if len(agent.buffer) >= rollout_len:
                agent.update()

        lam = float(getattr(agent, "lagrange_multiplier", float("nan")))
        step_at_ep.append(step_count)
        ep_rewards.append(ep_r)
        ep_maps.append(float(np.mean(ep_map_list)) if ep_map_list else 0.0)
        ep_costs.append(ep_c_sum / ep_n if ep_n > 0 else 0.0)
        ep_lambdas.append(lam)

        if len(ep_rewards) % 20 == 0:
            n   = min(20, len(ep_rewards))
            r20 = float(np.mean(ep_rewards[-n:]))
            m20 = float(np.mean(ep_maps[-n:]))
            c20 = float(np.mean(ep_costs[-n:]))
            print(f"  step {step_count:>7}  ep={len(ep_rewards):>4}"
                  f"  r={r20:+.2f}  MAP={m20:.1f}mmHg"
                  f"  cost/step={c20:.3f}  lambda={lam:.4f}")

    # ── save policy ───────────────────────────────────────────────────────────
    import torch
    actor_sd = agent.actor.state_dict()
    torch.save({"algo": agent_name, "actor": actor_sd}, save_path / "policy.pt")
    torch.save(actor_sd, save_path / "actor.pt")

    extra: dict = {}
    if hasattr(agent, "lagrange_multiplier"):
        extra["lagrange_multiplier"] = float(agent.lagrange_multiplier)
    if hasattr(agent, "cost_critic"):
        torch.save(agent.cost_critic.state_dict(), save_path / "cost_critic.pt")
        extra["cost_critic"] = "cost_critic.pt"

    meta = {
        "agent_name":        agent_name,
        "total_steps":       total_steps,
        "rollout_len":       rollout_len,
        "n_episodes":        len(ep_rewards),
        "mean_return_last20": float(np.mean(ep_rewards[-20:])) if ep_rewards else None,
        "mean_map_last20":   float(np.mean(ep_maps[-20:]))     if ep_maps    else None,
        "mean_cost_last20":  float(np.mean(ep_costs[-20:]))    if ep_costs   else None,
        **extra,
    }
    (save_path / "_meta.json").write_text(json.dumps(meta, indent=2))

    _save_training_plot(
        save_path   = save_path,
        agent_name  = agent_name,
        step_at_ep  = step_at_ep,
        ep_rewards  = ep_rewards,
        ep_maps     = ep_maps,
        ep_costs    = ep_costs,
        ep_lambdas  = ep_lambdas,
        cost_limit  = cost_limit,
    )

    print(f"  [saved] {save_path}/policy.pt  (ep={len(ep_rewards)})")
