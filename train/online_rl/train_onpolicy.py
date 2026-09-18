"""
train/train_onpolicy.py — Option-SMDP on-policy loop (PPO, TRPO, Lagrangian*).

TaCoS-aware: the agent outputs option o_k = (u_tanh, δt_tanh), where δt sets
intervention duration.

Key differences from vanilla on-policy training:
  - Action space: OPTION_ACTION_DIM = action_dim + 1 (includes δt)
  - State space:  SMDP_STATE_DIM  = state_dim  + 2 (includes t_remain, k_remain)
  - Reward is already multiplied by δt in env.step() (integral form)
  - Logging includes δt statistics to reflect learned timing
  - Lagrangian agents use a cost signal (same helper as legacy CPO-style code)
"""
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt

from data.config import state_dim, action_dim
from rl.environment import OPTION_ACTION_DIM


def compute_cost(phys_state_denorm: np.ndarray,
                 action_norm: np.ndarray,
                 action_scale: np.ndarray) -> float:
    """Scalar safety cost for Lagrangian objectives (high vaso, high lactate, low SpO2)."""
    a_raw = np.expm1(action_norm * action_scale)
    vaso = float(a_raw[1]) if len(a_raw) > 1 else 0.

    cost = 0.
    if vaso > 0.5:
        cost += 1.0
    if len(phys_state_denorm) > 5:
        lactate = phys_state_denorm[5]
        if lactate > 4.0:
            cost += 1.0
        spo2 = phys_state_denorm[0]
        if spo2 < 88:
            cost += 1.0
    return cost


def _extract_dt_from_option(option: np.ndarray, env) -> float:
    """Extract the decoded δt (hours) from an option vector."""
    dt_tanh = float(option[action_dim])
    return env.dt_min + (dt_tanh + 1.0) / 2.0 * (env.dt_max - env.dt_min)


def _denorm_phys_state(smdp_state: np.ndarray,
                       mean_np: np.ndarray,
                       std_np: np.ndarray) -> np.ndarray:
    """Extract physiological state from SMDP state and denormalize."""
    phys = smdp_state[:state_dim]
    return phys * std_np[:state_dim] + mean_np[:state_dim]


def _sample_init_state_norm(init_state_norm):
    init_arr = np.asarray(init_state_norm, dtype=np.float32)
    if init_arr.ndim == 2:
        init_arr = init_arr[np.random.randint(init_arr.shape[0])]
    return init_arr


def train_onpolicy_tacos(
    agent,
    env,
    patients: list | None,
    init_state_norm: np.ndarray | None = None,
    total_steps: int = 100_000,
    rollout_len: int = 2048,
    save_dir: str = "results",
    cluster_name: str = "",
    agent_name: str = "PPO",
    mean_np: np.ndarray | None = None,
    std_np: np.ndarray | None = None,
    action_scale_np: np.ndarray | None = None,
):
    """On-policy Option-SMDP training loop (PPO / TRPO / LagrangianPPO / LagrangianTRPO).

    Agent output: option = (u_tanh[3], dt_tanh[1]), 4-dim.
    mean_np / std_np / action_scale_np are used only by constrained agents to compute cost.
    """
    tag = f"[{cluster_name}/{agent_name}] " if cluster_name else f"[{agent_name}] "
    _CONSTRAINED_PREFIXES = {"LAGRANGIAN_PPO", "LAGRANGIAN_TRPO"}
    is_constrained = any(agent_name.upper().startswith(n) for n in _CONSTRAINED_PREFIXES)

    step_count = 0
    ep_rewards = []
    ep_sofas = []
    ep_dts = []
    ep_costs = []
    update_metrics = []

    print(f"\n{'=' * 60}")
    print(f"{tag}Option-SMDP Training: {total_steps} steps, rollout={rollout_len}")
    print(f"  state_dim={env.smdp_state_dim}  "
          f"action_dim={env.option_action_dim}  "
          f"dt=[{env.dt_min},{env.dt_max}]h")
    print(f"{'=' * 60}")

    while step_count < total_steps:
        if init_state_norm is not None:
            state = env.reset(init_state_norm=_sample_init_state_norm(init_state_norm))
        else:
            patient_pool = patients or [None]
            patient = random.choice(patient_pool)
            if patient is None:
                state = env.reset(patient=None,
                                  init_state_norm=np.zeros(state_dim, dtype=np.float32))
            else:
                state = env.reset(patient=patient)
        ep_reward = 0.
        ep_sofa_list = []
        ep_dt_list = []
        ep_cost = 0.
        done = False

        while not done and step_count < total_steps:
            if is_constrained:
                action, log_prob, value, cost_value = agent.select_action(state)
            else:
                action, log_prob, value = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            if is_constrained:
                phys_denorm = (
                    _denorm_phys_state(state, mean_np, std_np)
                    if mean_np is not None
                    else state[:state_dim]
                )
                u_norm = (action[:action_dim] + 1.0) / 2.0
                cost = compute_cost(
                    phys_denorm, u_norm,
                    action_scale_np if action_scale_np is not None else np.ones(action_dim),
                )
                agent.store_transition(
                    state, action, reward, cost,
                    next_state, float(done), log_prob, value, cost_value,
                )
                ep_cost += cost
            else:
                agent.store_transition(
                    state, action, reward,
                    next_state, float(done), log_prob, value,
                )

            state = next_state
            ep_reward += reward
            ep_sofa_list.append(info["sofa"])
            ep_dt_list.append(info["dt"])
            step_count += 1

            if len(agent.buffer) >= rollout_len:
                metrics = agent.update()
                update_metrics.append(metrics)

        ep_rewards.append(ep_reward)
        ep_sofas.append(np.mean(ep_sofa_list) if ep_sofa_list else 0.)
        ep_dts.append(np.mean(ep_dt_list) if ep_dt_list else 0.)
        if is_constrained:
            ep_costs.append(ep_cost)

        if len(ep_rewards) % 20 == 0:
            recent_r = np.mean(ep_rewards[-20:])
            recent_s = np.mean(ep_sofas[-20:])
            recent_dt = np.mean(ep_dts[-20:])
            cost_str = f" cost={np.mean(ep_costs[-20:]):.1f}" if ep_costs else ""

            det_action = agent.select_action(state, deterministic=True)
            det_dt = _extract_dt_from_option(det_action, env)

            print(f"{tag}Step {step_count:6d} | ep={len(ep_rewards):4d} | "
                  f"reward={recent_r:.1f} | sofa={recent_s:.1f} | "
                  f"mean_δt={recent_dt:.2f}h | det_δt={det_dt:.2f}h{cost_str}")

    if len(agent.buffer) > 0:
        agent.update()

    os.makedirs(save_dir, exist_ok=True)
    agent.save(f"{save_dir}/policy.pt")

    # ── Save training data ────────────────────────────────────
    np.savez(
        f"{save_dir}/curves.npz",
        rewards=np.array(ep_rewards),
        sofas=np.array(ep_sofas),
        dts=np.array(ep_dts),
        costs=np.array(ep_costs) if ep_costs else np.array([]),
    )

    # ── Training curves ───────────────────────────────────────
    n_plots = 4 if is_constrained else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    axes[0].plot(ep_rewards)
    axes[0].set_title(f"{tag}Episode Reward (integrated)")
    axes[0].set_xlabel("Episode")

    axes[1].plot(ep_sofas)
    axes[1].set_title(f"{tag}Mean SOFA")
    axes[1].set_xlabel("Episode")

    axes[2].plot(ep_dts)
    axes[2].axhline(y=env.dt_min, ls="--", c="r", lw=0.8, label=f"dt_min={env.dt_min}h")
    axes[2].axhline(y=env.dt_max, ls="--", c="g", lw=0.8, label=f"dt_max={env.dt_max}h")
    axes[2].set_title(f"{tag}Mean δt per Episode (learned timing)")
    axes[2].set_xlabel("Episode")
    axes[2].legend(fontsize=8)

    if is_constrained:
        axes[3].plot(ep_costs)
        axes[3].axhline(y=agent.cost_limit, ls="--", c="r", lw=0.8, label=f"limit={agent.cost_limit}")
        axes[3].set_title(f"{tag}Episode Cost")
        axes[3].set_xlabel("Episode")
        axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training.png", dpi=150)
    plt.close()

    with open(f"{save_dir}/_meta.json", "w") as f:
        json.dump(
            {
                "episode_rewards": [float(x) for x in ep_rewards],
                "episode_sofas": [float(x) for x in ep_sofas],
                "episode_dts": [float(x) for x in ep_dts],
                "episode_costs": [float(x) for x in ep_costs] if ep_costs else [],
                "total_steps": int(total_steps),
                "rollout_len": int(rollout_len),
                "n_patients": int(len(patients or [])),
                "cluster_name": cluster_name,
                "agent_name": agent_name,
                "dt_range": [env.dt_min, env.dt_max],
                "total_time_h": env.total_time_h,
                "max_steps": env.max_steps,
            },
            f,
            indent=2,
        )

    return ep_rewards, ep_sofas, ep_dts
