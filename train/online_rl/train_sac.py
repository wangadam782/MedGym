"""train/train_sac.py — unified SAC training loop (fixed-dt and time-adaptive).

tag is inferred from env.use_dt:
  use_dt=True  → tag="vardt"   saves sac_vardt.pt / sac_vardt_curves.npz
  use_dt=False → tag="fixdt"   saves sac_fixdt.pt / sac_fixdt_curves.npz
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from data.config import state_dim
from rl.sac import ReplayBuffer


def _reset_env(env, patient=None, init_state_norm=None):
    if init_state_norm is not None:
        init_arr = np.asarray(init_state_norm, dtype=np.float32)
        if init_arr.ndim == 2:
            init_arr = init_arr[np.random.randint(init_arr.shape[0])]
        return env.reset(init_state_norm=init_arr)
    if patient is not None:
        return env.reset(patient=patient)
    return env.reset(init_state_norm=np.zeros(state_dim, dtype=np.float32))


def _smooth(vals, window=20):
    import pandas as pd
    s = pd.Series(vals).rolling(window, min_periods=1, center=True).mean()
    return s.to_numpy()


def train_sac(
    agent,
    env,
    patient,
    init_state_norm=None,
    total_steps: int = 100_000,
    batch_size:  int = 512,
    start_steps: int = 20_000,
    save_dir:    str = "results",
    tag:         str = None,
):
    """Unified SAC training loop for fixed-dt and time-adaptive environments.

    tag : "vardt" / "fixdt" — inferred from env.use_dt when None.
    """
    if tag is None:
        tag = "vardt" if env.use_dt else "fixdt"

    act_dim    = env.option_action_dim
    buf        = ReplayBuffer(200_000)
    step_count = 0
    ep_rewards = []
    ep_sofas   = []
    ep_dts     = []

    print(f"\n{'='*55}")
    print(f"SAC Training [{tag}]: {total_steps} steps")
    print(f"  state_dim={env.smdp_state_dim}  action_dim={act_dim}  use_dt={env.use_dt}")
    if env.use_dt:
        print(f"  dt=[{env.dt_min},{env.dt_max}]h")
    else:
        print(f"  fixed_dt={env.fixed_dt}h")
    print(f"  start_steps={start_steps}  batch_size={batch_size}")
    print(f"{'='*55}")

    while step_count < total_steps:
        state        = _reset_env(env, patient=patient, init_state_norm=init_state_norm)
        ep_reward    = 0.
        ep_sofa_list = []
        ep_dt_list   = []
        done         = False

        while not done:
            if step_count < start_steps:
                action = np.random.uniform(-1., 1., size=act_dim).astype(np.float32)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            buf.push(state, action, reward, next_state, float(done))

            state        = next_state
            ep_reward   += reward
            ep_sofa_list.append(info["sofa"])
            ep_dt_list.append(info["dt"])
            step_count  += 1

            if step_count >= start_steps:
                agent.update(buf, batch_size)

            if step_count % 5_000 == 0 and step_count >= start_steps:
                test_state  = _reset_env(env, patient=patient, init_state_norm=init_state_norm)
                test_action = agent.select_action(test_state, deterministic=True)
                recent_r    = np.mean(ep_rewards[-20:]) if ep_rewards else 0
                if env.use_dt:
                    test_dt = (env.dt_min
                               + (float(test_action[-1]) + 1.0) / 2.0
                               * (env.dt_max - env.dt_min))
                    print(
                        f"Step {step_count:7d}/{total_steps} | "
                        f"reward={recent_r:.2f} | "
                        f"sofa={np.mean(ep_sofas[-20:] or [0]):.1f} | "
                        f"mean_δt={np.mean(ep_dts[-20:] or [0]):.2f}h | "
                        f"det_dt={test_dt:.2f}h  det_u={test_action[:-1].round(2)}"
                    )
                else:
                    print(
                        f"Step {step_count:7d}/{total_steps} | "
                        f"reward={recent_r:.2f} | "
                        f"sofa={np.mean(ep_sofas[-20:] or [0]):.1f} | "
                        f"alpha={agent.alpha:.4f} | "
                        f"det_u={test_action.round(3)}"
                    )

        ep_rewards.append(ep_reward)
        ep_sofas.append(float(np.mean(ep_sofa_list)))
        ep_dts.append(float(np.mean(ep_dt_list)))

    os.makedirs(save_dir, exist_ok=True)

    model_path  = f"{save_dir}/policy.pt"
    curves_path = f"{save_dir}/curves.npz"
    agent.save(model_path)
    np.savez(
        curves_path,
        ep_rewards = np.array(ep_rewards, dtype=np.float32),
        ep_sofas   = np.array(ep_sofas,   dtype=np.float32),
        ep_dts     = np.array(ep_dts,     dtype=np.float32),
        use_dt     = np.bool_(env.use_dt),
    )
    print(f"[SAC] Curves saved → {curves_path}")

    n_axes = 3 if env.use_dt else 2
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 4))
    if n_axes == 2:
        axes = list(axes)
    fig.suptitle(f"SAC Training [{tag}]", fontsize=12, fontweight="bold")

    axes[0].plot(ep_rewards, alpha=0.5)
    axes[0].plot(_smooth(ep_rewards), lw=2, color="tab:blue")
    axes[0].set_title("Episode Reward")

    axes[1].plot(ep_sofas, alpha=0.5, color="tab:red")
    axes[1].plot(_smooth(ep_sofas), lw=2, color="tab:red")
    axes[1].set_title("Mean SOFA")

    if env.use_dt:
        axes[2].plot(ep_dts, alpha=0.5, color="tab:green")
        axes[2].plot(_smooth(ep_dts), lw=2, color="tab:green")
        axes[2].axhline(y=env.dt_min, color="r", ls="--", lw=1.0,
                        label=f"dt_min={env.dt_min}h")
        axes[2].axhline(y=env.dt_max, color="g", ls="--", lw=1.0,
                        label=f"dt_max={env.dt_max}h")
        axes[2].set_title("Mean δt")
        axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training.png", dpi=150)
    plt.close()

    return ep_rewards, ep_sofas, ep_dts


def train_cluster_sac(
    agent,
    env,
    cluster_patients: list,
    total_steps: int = 60_000,
    batch_size:  int = 512,
    start_steps: int = 12_000,
    save_dir:    str = "results",
    tag:         str = None,
):
    """Unified cluster SAC training (fixed-dt and time-adaptive).

    Randomly samples a patient at the start of each episode.
    tag : "vardt" / "fixdt" — inferred from env.use_dt when None.
    """
    if tag is None:
        tag = "vardt" if env.use_dt else "fixdt"

    act_dim    = env.option_action_dim
    buf        = ReplayBuffer(200_000)
    step_count = 0
    ep_rewards = []
    ep_sofas   = []
    ep_dts     = []

    print(f"\n{'='*60}")
    print(f"Cluster SAC Training [{tag}]: {total_steps} steps  "
          f"n_patients={len(cluster_patients)}")
    print(f"  state_dim={env.smdp_state_dim}  action_dim={act_dim}  use_dt={env.use_dt}")
    print(f"  start_steps={start_steps}  batch_size={batch_size}")
    print(f"{'='*60}")

    while step_count < total_steps:
        patient      = random.choice(cluster_patients)
        state        = env.reset(patient=patient)
        ep_reward    = 0.
        ep_sofa_list = []
        ep_dt_list   = []
        done         = False

        while not done:
            if step_count < start_steps:
                action = np.random.uniform(-1., 1., size=act_dim).astype(np.float32)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            buf.push(state, action, reward, next_state, float(done))

            state        = next_state
            ep_reward   += reward
            ep_sofa_list.append(info["sofa"])
            ep_dt_list.append(info["dt"])
            step_count  += 1

            if step_count >= start_steps:
                agent.update(buf, batch_size)

            if step_count % 5_000 == 0 and step_count >= start_steps:
                test_patient = random.choice(cluster_patients)
                test_state   = env.reset(patient=test_patient)
                test_action  = agent.select_action(test_state, deterministic=True)
                recent_r     = np.mean(ep_rewards[-20:]) if ep_rewards else 0
                if env.use_dt:
                    test_dt = (env.dt_min
                               + (float(test_action[-1]) + 1.0) / 2.0
                               * (env.dt_max - env.dt_min))
                    print(
                        f"Step {step_count:6d} | "
                        f"ep_r={recent_r:.2f} | "
                        f"sofa={np.mean(ep_sofas[-20:] or [0]):.1f} | "
                        f"mean_δt={np.mean(ep_dts[-20:] or [0]):.2f}h | "
                        f"det_δt={test_dt:.2f}h"
                    )
                else:
                    print(
                        f"Step {step_count:6d} | "
                        f"ep_r={recent_r:.2f} | "
                        f"sofa={np.mean(ep_sofas[-20:] or [0]):.1f} | "
                        f"alpha={agent.alpha:.4f} | "
                        f"det_u={test_action.round(3)}"
                    )

        ep_rewards.append(ep_reward)
        ep_sofas.append(float(np.mean(ep_sofa_list)))
        ep_dts.append(float(np.mean(ep_dt_list)))

    os.makedirs(save_dir, exist_ok=True)
    agent.save(f"{save_dir}/policy.pt")
    np.savez(
        f"{save_dir}/curves.npz",
        ep_rewards = np.array(ep_rewards, dtype=np.float32),
        ep_sofas   = np.array(ep_sofas,   dtype=np.float32),
        ep_dts     = np.array(ep_dts,     dtype=np.float32),
        use_dt     = np.bool_(env.use_dt),
    )

    n_axes = 3 if env.use_dt else 2
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 4))
    if n_axes == 2:
        axes = list(axes)
    fig.suptitle(f"Cluster SAC Training [{tag}]", fontsize=12, fontweight="bold")
    axes[0].plot(ep_rewards); axes[0].set_title("Episode Reward"); axes[0].set_xlabel("Episode")
    axes[1].plot(ep_sofas);   axes[1].set_title("Mean SOFA");     axes[1].set_xlabel("Episode")
    if env.use_dt:
        axes[2].plot(ep_dts)
        axes[2].axhline(y=env.dt_min, ls="--", c="r", lw=0.8,
                        label=f"dt_min={env.dt_min}h")
        axes[2].axhline(y=env.dt_max, ls="--", c="g", lw=0.8,
                        label=f"dt_max={env.dt_max}h")
        axes[2].set_title("Mean δt per Episode"); axes[2].set_xlabel("Episode")
        axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training.png", dpi=150)
    plt.close()

    return ep_rewards, ep_sofas, ep_dts


# Backward-compatible aliases
def train_sac_time_adaptive(agent, env, patient, **kwargs):
    return train_sac(agent, env, patient, tag="vardt", **kwargs)


def train_cluster_sac_time_adaptive(agent, env, cluster_patients, **kwargs):
    return train_cluster_sac(agent, env, cluster_patients, tag="vardt", **kwargs)
