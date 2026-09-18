"""train/train_ppo.py — On-policy training loop for PPO and TRPO.

Both agents share the same interface:
  - select_action(state)                → (action, log_prob, value)
  - select_action(state, deterministic) → action
  - store_transition(state, action, reward, next_state, done, log_prob, value)
  - update()                            → metrics dict
  - save(path) / load(path)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _smooth(vals, window: int = 20):
    import pandas as pd
    s = pd.Series(vals).rolling(window, min_periods=1, center=True).mean()
    return s.to_numpy()


def _sample_init_state_norm(init_state_norm):
    init_arr = np.asarray(init_state_norm, dtype=np.float32)
    if init_arr.ndim == 2:
        init_arr = init_arr[np.random.randint(init_arr.shape[0])]
    return init_arr


def train_ppo(
    agent,
    env,
    patient,
    init_state_norm=None,
    total_steps:    int = 200_000,
    rollout_len:    int = 2_048,
    save_dir:       str = "results",
    tag:            str = None,
    algo_name:      str = None,
):
    """Train a PPO or TRPO agent in the ICU Option-SMDP environment."""
    if tag is None:
        tag = "vardt" if env.use_dt else "fixdt"
    if algo_name is None:
        algo_name = type(agent).__name__.lower()

    os.makedirs(save_dir, exist_ok=True)

    ep_rewards: list[float] = []
    ep_sofas:   list[float] = []
    ep_dts:     list[float] = []

    print(f"\n{'='*58}")
    print(f"  {algo_name.upper()} Training [{tag}]  —  {total_steps} steps")
    print(f"  rollout_len={rollout_len}")
    print(f"  state={env.smdp_state_dim}  act={env.option_action_dim}  "
          f"use_dt={env.use_dt}")
    print(f"{'='*58}")

    step_count = 0
    ep_reward, ep_sofa_list, ep_dt_list = 0.0, [], []

    if init_state_norm is not None:
        state = env.reset(init_state_norm=_sample_init_state_norm(init_state_norm))
    elif patient is not None:
        state = env.reset(patient=patient)
    else:
        state = env.reset(init_state_norm=np.zeros(
            env.smdp_state_dim - 2, dtype=np.float32))
    done = False

    while step_count < total_steps:
        # Collect rollout
        rollout_steps = 0
        while rollout_steps < rollout_len and step_count < total_steps:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward,
                                   next_state, done, log_prob, value)

            ep_reward     += reward
            ep_sofa_list.append(info["sofa"])
            ep_dt_list.append(info["dt"])
            state      = next_state
            step_count += 1
            rollout_steps += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_sofas.append(float(np.mean(ep_sofa_list)) if ep_sofa_list else 0.0)
                ep_dts.append(float(np.mean(ep_dt_list)) if ep_dt_list else 0.0)

                if len(ep_rewards) % 20 == 0:
                    print(
                        f"  Step {step_count:7d}/{total_steps} | "
                        f"ep={len(ep_rewards):4d} | "
                        f"r={np.mean(ep_rewards[-20:]):.2f} | "
                        f"sofa={np.mean(ep_sofas[-20:]):.1f} | "
                        f"dt={np.mean(ep_dts[-20:]):.2f}h"
                    )

                ep_reward, ep_sofa_list, ep_dt_list = 0.0, [], []

                if init_state_norm is not None:
                    state = env.reset(init_state_norm=_sample_init_state_norm(init_state_norm))
                elif patient is not None:
                    state = env.reset(patient=patient)
                else:
                    state = env.reset(init_state_norm=np.zeros(
                        env.smdp_state_dim - 2, dtype=np.float32))
                done = False

        # Update policy
        agent.update()

    # Save remaining episode if mid-episode
    if ep_sofa_list:
        ep_rewards.append(ep_reward)
        ep_sofas.append(float(np.mean(ep_sofa_list)))
        ep_dts.append(float(np.mean(ep_dt_list)))

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(save_dir, f"policy.pt")
    agent.save(model_path)

    # ── Save curves ───────────────────────────────────────────────────────────
    curves_path = os.path.join(save_dir, f"curves.npz")
    np.savez(
        curves_path,
        ep_rewards = np.array(ep_rewards, dtype=np.float32),
        ep_sofas   = np.array(ep_sofas,   dtype=np.float32),
        ep_dts     = np.array(ep_dts,     dtype=np.float32),
        use_dt     = np.bool_(env.use_dt),
        algo       = np.str_(algo_name),
    )
    print(f"\n[{algo_name.upper()}] Curves  → {curves_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{algo_name.upper()} Training [{tag}]", fontsize=12, fontweight="bold")

    axes[0].plot(ep_rewards, alpha=0.4)
    axes[0].plot(_smooth(ep_rewards), lw=2, color="tab:blue")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")

    axes[1].plot(ep_sofas, alpha=0.4, color="tab:red")
    axes[1].plot(_smooth(ep_sofas), lw=2, color="tab:red")
    axes[1].set_title("Mean SOFA")
    axes[1].set_xlabel("Episode")

    axes[2].plot(ep_dts, alpha=0.4, color="tab:green")
    axes[2].plot(_smooth(ep_dts), lw=2, color="tab:green")
    if env.use_dt:
        axes[2].axhline(y=env.dt_min, color="r", ls="--", lw=0.8,
                        label=f"dt_min={env.dt_min}h")
        axes[2].axhline(y=env.dt_max, color="g", ls="--", lw=0.8,
                        label=f"dt_max={env.dt_max}h")
        axes[2].legend(fontsize=8)
    else:
        axes[2].axhline(y=env.fixed_dt, color="gray", ls="--", lw=1.0,
                        label=f"fixed_dt={env.fixed_dt}h")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Mean δt per Episode")
    axes[2].set_xlabel("Episode")

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"training.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[{algo_name.upper()}] Figure  → {fig_path}")

    return ep_rewards, ep_sofas, ep_dts
