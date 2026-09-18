"""train_cql.py — Batch offline DDPG+CQL training over collected NPZ datasets.

This script trains one offline policy per dataset tag. It is used for both:
  - CQL, by setting --cql_alpha > 0
  - DQN-style training, by setting --cql_alpha 0.0

Expected input datasets:
  results/offline/datasets/
    pid_<pid>_fixdt_dataset.npz
    trpo_pop_fixdt_dataset.npz

Saved outputs:
  <save_dir>/
    models/
      pid_<pid>_fixdt/
        actor_final.pt
        critic_final.pt
      trpo_pop_fixdt/
        actor_final.pt
        critic_final.pt
    logs/
      <tag>_trainlog.npz
    training_summary.csv

The standard offline training workflow reads only the collected NPZ datasets and
does not require `data/mimic_pinn_v4_filtered.csv`.

Patient-list driven individual training:
  Instead of manually passing many --tags values, pass --patient_ids_file
  to generate individual tags such as pid_<pid>_fixdt from a CSV/TXT
  patient list. --tags still takes precedence when both are provided.

Optional environment evaluation:
  If --eval_in_env is enabled, the script can build a patient PINN environment
  for per-patient datasets. In the released no-MIMIC workflow, use
  --no_mimic_csv and make sure each patient PINN directory contains
  init_state_norm.npy. A processed MIMIC CSV can still be passed explicitly via
  --csv as a development-only fallback when init_state_norm.npy is unavailable.

Example: individual CQL
    python3 scripts/offline/train_cql.py \\
        --dataset_dir results/offline/datasets \\
        --save_dir results/offline/policies/individual/cql \\
        --epochs 5 \\
        --batch_size 256 \\
        --cql_alpha 5.0 \\
        --tags pid_200325_fixdt pid_201046_fixdt pid_201101_fixdt


Example: individual CQL from a patient list
    python3 scripts/offline/train_cql.py \\
        --dataset_dir results/offline/datasets \\
        --save_dir results/offline/policies/individual/cql \\
        --epochs 5 \\
        --batch_size 256 \\
        --cql_alpha 5.0 \\
        --patient_ids_file checkpoints-cohort/cohort_1/cohort_1_training.csv \\
        --dt_mode fixdt

Example: population CQL
    python3 scripts/offline/train_cql.py \\
        --dataset_dir results/offline/datasets \\
        --save_dir results/offline/policies/population/cql \\
        --epochs 5 \\
        --batch_size 256 \\
        --cql_alpha 5.0 \\
        --tags trpo_pop_fixdt

Example: DQN-style training
    python3 scripts/offline/train_cql.py \\
        --dataset_dir results/offline/datasets \\
        --save_dir results/offline/policies/individual/dqn \\
        --epochs 5 \\
        --batch_size 256 \\
        --cql_alpha 0.0 \\
        --tags pid_200325_fixdt pid_201046_fixdt pid_201101_fixdt
"""

import argparse
import copy
import os
import random
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OFFLINE_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_OFFLINE_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFLINE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pinn_bundle import load_patient_ids_from_file

# ══════════════════════════════════════════════════════════════
# 0. Global hyper-parameters (overridable via argparse)
# ══════════════════════════════════════════════════════════════
GAMMA          = 0.99
TAU            = 0.001
BATCH_SIZE     = 256
NUM_EPOCHS     = 4_000
HIDDEN_SIZE    = 256
CLIP_REWARD    = True
REWARD_CLIP_LO = -8
REWARD_CLIP_HI = -3

NUM_CANDIDATES = 12
NOISE_STD      = 0.1
CQL_ALPHA      = 5.0

SAVE_INTERVAL  = 500
EVAL_INTERVAL  = 200

STATE_DIM  = 6
ACTION_DIM = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# 1. Networks (identical to train_ddpg_cql.py)
# ══════════════════════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_SIZE):
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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_SIZE):
        super().__init__()
        self.fc1    = nn.Linear(state_dim + action_dim, hidden)
        self.bn1    = nn.BatchNorm1d(hidden)
        self.fc2    = nn.Linear(hidden, hidden)
        self.bn2    = nn.BatchNorm1d(hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        return self.fc_out(x)


def soft_update(source, target, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ══════════════════════════════════════════════════════════════
# 2. Offline dataset loader
# ══════════════════════════════════════════════════════════════
class OfflineDataset:
    def __init__(self, npz_path, clip_reward=CLIP_REWARD):
        data = np.load(npz_path, allow_pickle=False)

        self.S    = torch.tensor(data["transitions/s"],    dtype=torch.float32, device=DEVICE)
        self.A    = torch.tensor(data["transitions/a"],    dtype=torch.float32, device=DEVICE)
        r_raw     = data["transitions/r"].astype(np.float32)
        if clip_reward:
            r_raw = np.clip(r_raw, REWARD_CLIP_LO, REWARD_CLIP_HI)
        self.R    = torch.tensor(r_raw,                    dtype=torch.float32, device=DEVICE)
        self.NS   = torch.tensor(data["transitions/ns"],   dtype=torch.float32, device=DEVICE)
        self.DONE = torch.tensor(data["transitions/done"], dtype=torch.float32, device=DEVICE)

        self.sofa_all = (data["meta/sofa"].astype(np.float32)
                         if "meta/sofa" in data else np.array([]))
        self.ep_reward = (data["meta/ep_reward_sum"].astype(np.float32)
                          if "meta/ep_reward_sum" in data else np.array([]))

        self.N = len(self.R)
        sd = self.S.shape[1] if self.S.ndim > 1 else STATE_DIM
        ad = self.A.shape[1] if self.A.ndim > 1 else ACTION_DIM
        print(f"  [Dataset] {self.N:,} transitions | s={sd}  a={ad} | "
              f"r∈[{self.R.min():.2f},{self.R.max():.2f}] "
              f"mean={self.R.mean():.3f}", flush=True)

    def sample(self, batch_size):
        idx = torch.randint(0, self.N, (batch_size,), device=DEVICE)
        return (self.S[idx], self.A[idx],
                self.R[idx].unsqueeze(1), self.NS[idx],
                self.DONE[idx].unsqueeze(1))

    @property
    def baseline(self):
        return {
            "r_mean":    float(self.R.mean()),
            "r_std":     float(self.R.std()),
            "sofa_mean": float(self.sofa_all.mean()) if len(self.sofa_all) > 0 else float("nan"),
        }


# ══════════════════════════════════════════════════════════════
# 3. CQL update step (identical to train_ddpg_cql.py)
# ══════════════════════════════════════════════════════════════
def cql_update_step(
    actor, critic, target_actor, target_critic,
    opt_actor, opt_critic,
    S, A, R, NS, D,
    n_candidates, noise_std, cql_alpha, gamma,
):
    B = S.shape[0]
    sd = S.shape[1]
    ad = A.shape[1]

    # ── Bellman target ─────────────────────────────
    with torch.no_grad():
        next_A_target = target_actor(NS)
        y = R + gamma * target_critic(NS, next_A_target) * (1 - D)

    # ── Critic: data Q ─────────────────────────────
    Q_data = critic(S, A)
    loss_mse = F.mse_loss(Q_data, y)

    # ── CQL penalty ────────────────────────────────
    with torch.no_grad():
        actor_A = actor(S)

    cand_A = actor_A.unsqueeze(1) + torch.randn(
        B, n_candidates, ad, device=DEVICE
    ) * noise_std
    cand_A = cand_A.clamp(-1.0, 1.0)

    S_exp = S.unsqueeze(1).expand(-1, n_candidates, -1).reshape(-1, sd)
    A_exp = cand_A.reshape(-1, ad)

    Q_cand = critic(S_exp, A_exp).view(B, n_candidates)

    logsumexp_q = torch.logsumexp(Q_cand, dim=1)
    cql_penalty = (logsumexp_q - Q_data.squeeze(1)).mean()

    # ── Critic loss ────────────────────────────────
    loss_critic = loss_mse + cql_alpha * cql_penalty

    opt_critic.zero_grad()
    loss_critic.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opt_critic.step()

    # ── Actor loss ────────────────────────────────
    pi_A = actor(S)
    loss_actor = -critic(S, pi_A).mean()

    opt_actor.zero_grad()
    loss_actor.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_actor.step()

    # ── Target update ─────────────────────────────
    soft_update(actor, target_actor, TAU)
    soft_update(critic, target_critic, TAU)

    return {
        "loss_critic": float(loss_critic.item()),
        "loss_mse": float(loss_mse.item()),
        "cql_penalty": float(cql_penalty.item()),
        "loss_actor": float(loss_actor.item()),
        "q_data_mean": float(Q_data.mean().item()),
    }

# ══════════════════════════════════════════════════════════════
# 4. Optional PINN environment evaluation
# ══════════════════════════════════════════════════════════════
def build_pinn_env(pinn_dir, csv_path, pid, args):
    """Build a PINN evaluation environment for one patient.

    ``csv_path`` may be ``None`` when ``init_state_norm.npy`` exists next to ``pinn.pt``.
    """
    from data.config import state_dim as sd, action_dim as ad
    from models      import PINN
    from rl          import ICUEnvironment

    from pinn_bundle import load_init_and_scales_for_patient

    init_norm, pinn_path, mean_np, std_np, ascl_np, state_min_np, state_max_np = (
        load_init_and_scales_for_patient(Path(pinn_dir), pid, csv_path)
    )

    pinn = PINN(sd, ad).to(DEVICE)
    pinn.load_state_dict(torch.load(str(pinn_path), map_location=DEVICE))
    pinn.eval()

    TOTAL_TIME_H = 96.0
    K            = args.k
    fixed_dt     = TOTAL_TIME_H / K

    env = ICUEnvironment(
        pinn_model      = pinn,
        mean_np         = mean_np,
        std_np          = std_np,
        action_min_norm = np.zeros(ad, dtype=np.float32),
        action_max_norm = np.ones(ad,  dtype=np.float32),
        state_min       = state_min_np,
        state_max       = state_max_np,
        action_scale_np = ascl_np,
        max_steps       = K,
        dt_min          = 0.5,
        dt_max          = 36.0,
        total_time_h    = TOTAL_TIME_H,
        use_dt          = False,
        fixed_dt        = fixed_dt,
        use_lac_penalty = True,
        lactate_threshold=args.lactate_threshold,
    )
    return env, init_norm


def eval_in_env(actor, env, init_state_norm, n_ep=5, state_dim=STATE_DIM):
    actor.eval()
    all_r, all_sofa = [], []
    x0 = np.asarray(init_state_norm, dtype=np.float32)
    for _ in range(n_ep):
        state = env.reset(init_state_norm=x0)
        ep_r  = 0.0
        steps = 0
        sofa_list = []
        while True:
            s_t = torch.tensor(state[:state_dim], dtype=torch.float32,
                               device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = actor(s_t).squeeze(0).cpu().numpy()
            next_state, r, done, info = env.step(action)
            ep_r += r
            steps += 1
            sofa_list.append(info["sofa"])
            state = next_state
            if done:
                break
        all_r.append(ep_r / max(steps, 1))
        all_sofa.append(float(np.mean(sofa_list)))
    actor.train()
    return {"reward": float(np.mean(all_r)), "sofa": float(np.mean(all_sofa))}


# ══════════════════════════════════════════════════════════════
# 5. Single-dataset training function
# ══════════════════════════════════════════════════════════════
def train_one(npz_path, model_dir, log_dir, tag, args,
              pinn_env=None, pinn_init_norm=None):
    """Train one CQL model on a single NPZ dataset. Returns (log, eval_log)."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Training: {tag}", flush=True)
    print(f"  NPZ : {npz_path}", flush=True)

    dataset = OfflineDataset(npz_path, clip_reward=args.clip_reward)
    sd = dataset.S.shape[1]
    ad = dataset.A.shape[1]


    actor         = Actor(sd, ad, args.hidden).to(DEVICE)
    critic        = Critic(sd, ad, args.hidden).to(DEVICE)
    target_actor  = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)
    opt_actor     = optim.Adam(actor.parameters(),  lr=args.lr)
    opt_critic    = optim.Adam(critic.parameters(), lr=args.lr)

    log = {k: [] for k in ["step", "loss_critic", "loss_mse",
                            "cql_penalty", "loss_actor", "q_data_mean",
                            "supp_pen_c", "supp_pen_a"]}
    eval_log = {"step": [], "reward": [], "sofa": []}

    baseline = dataset.baseline
    print(f"  Baseline: r={baseline['r_mean']:.4f}  sofa={baseline['sofa_mean']:.3f}",
          flush=True)
    print(f"  Epochs={args.epochs}  batch={args.batch_size}  "
          f"cql_alpha={args.cql_alpha}  n_cand={args.n_candidates}", flush=True)
    print(f"{'='*60}", flush=True)

    for step in range(1, args.epochs + 1):
        S, A, R, NS, D = dataset.sample(args.batch_size)
        m = cql_update_step(
            actor, critic, target_actor, target_critic,
            opt_actor, opt_critic,
            S, A, R, NS, D,
            n_candidates=args.n_candidates,
            noise_std=args.noise_std,
            cql_alpha=args.cql_alpha,
            gamma=args.gamma,
        )
        log["step"].append(step)
        for k, v in m.items():
            log[k].append(v)

        if step % SAVE_INTERVAL == 0 or step == 1:
            print(
                f"  [{tag}] step={step:6d} | "
                f"critic={m['loss_critic']:.4f} | mse={m['loss_mse']:.4f} | "
                f"cql={m['cql_penalty']:.4f} | actor={m['loss_actor']:.4f} | "
                f"Q={m['q_data_mean']:.3f}",
                flush=True,
            )

        if step % args.eval_interval == 0 and pinn_env is not None and pinn_init_norm is not None:
            ev = eval_in_env(actor, pinn_env, pinn_init_norm,
                             n_ep=args.eval_episodes, state_dim=sd)
            eval_log["step"].append(step)
            eval_log["reward"].append(ev["reward"])
            eval_log["sofa"].append(ev["sofa"])
            dr = ev["reward"] - baseline["r_mean"]
            ds = ev["sofa"]   - baseline["sofa_mean"]
            print(
                f"    [EVAL] r={ev['reward']:.3f} ({dr:+.3f}) | "
                f"sofa={ev['sofa']:.2f} ({ds:+.2f})",
                flush=True,
            )

    # Save models
    torch.save(actor.state_dict(),  os.path.join(model_dir, "actor_final.pt"))
    torch.save(critic.state_dict(), os.path.join(model_dir, "critic_final.pt"))
    print(f"  [Saved] {model_dir}/actor_final.pt", flush=True)

    # Save training log
    log_data = {f"log/{k}": np.array(v) for k, v in log.items()}
    log_data.update({f"eval/{k}": np.array(v) for k, v in eval_log.items()})
    log_data["info/baseline_r_mean"]    = np.array([baseline["r_mean"]])
    log_data["info/baseline_sofa_mean"] = np.array([baseline["sofa_mean"]])
    log_data["info/tag"] = np.array([tag])
    np.savez_compressed(os.path.join(log_dir, f"{tag}_trainlog.npz"), **log_data)

    return log, eval_log, baseline


# ══════════════════════════════════════════════════════════════
# 6. Dataset discovery
# ══════════════════════════════════════════════════════════════
def discover_datasets(dataset_dir):
    """Scan dataset_dir for NPZ files and return sorted list of (tag, path).

    Expected filenames:
        pid_{id}_fixdt_dataset.npz
        pid_{id}_vardt_dataset.npz
        trpo_pop_fixdt_dataset.npz
        trpo_pop_vardt_dataset.npz
    """
    from pathlib import Path
    items = []
    ddir = Path(dataset_dir)
    for npz in sorted(ddir.glob("*.npz")):
        name = npz.stem  # e.g. 'pid_200325_fixdt_dataset'
        # derive a clean tag: strip trailing '_dataset'
        tag = name.replace("_dataset", "")
        items.append((tag, str(npz)))
    return items


def parse_pid_from_tag(tag):
    """Extract integer patient ID from a tag like 'pid_200325_fixdt', else None."""
    parts = tag.split("_")
    if parts[0] == "pid" and len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def dt_tags_from_mode(dt_mode):
    """Return dt suffixes corresponding to a requested dt_mode."""
    tags = []
    if dt_mode in ("fixdt", "both"):
        tags.append("fixdt")
    if dt_mode in ("vardt", "both"):
        tags.append("vardt")
    return tags


def tags_from_patient_ids_file(patient_ids_file, dt_mode):
    """Generate per-patient dataset tags from a patient ID file."""
    pids = load_patient_ids_from_file(patient_ids_file)
    dt_suffixes = dt_tags_from_mode(dt_mode)
    return {f"pid_{pid}_{dt}" for pid in pids for dt in dt_suffixes}


# ══════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Batch DDPG+CQL training over multiple offline datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_dir",  required=False,
                   default="results/offline/datasets",
                   help="Directory containing *_dataset.npz files")
    p.add_argument("--save_dir",     required=False,
                   default="results/offline/policies/individual/cql",
                   help="Root output directory")
    p.add_argument("--glb_scales",
                   default="results/pinn/population/scales.npy")
    p.add_argument("--tags",         nargs="*", default=None,
                   help="If set, only train on datasets whose tag matches one of these strings")
    p.add_argument(
        "--patient_ids_file",
        type=str,
        default=None,
        help=(
            "Optional pid-only CSV/TXT file used to generate individual dataset "
            "tags such as pid_<pid>_fixdt. Ignored when --tags is provided."
        ),
    )
    p.add_argument(
        "--dt_mode",
        type=str,
        default="fixdt",
        choices=["fixdt", "vardt", "both"],
        help="DT mode used when generating tags from --patient_ids_file.",
    )

    # Network
    p.add_argument("--hidden",        type=int,   default=HIDDEN_SIZE)
    p.add_argument("--lr",            type=float, default=3e-5)

    # Training
    p.add_argument("--epochs",        type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--gamma",         type=float, default=GAMMA)
    p.add_argument("--clip_reward",   action="store_true", default=CLIP_REWARD)

    # CQL
    p.add_argument("--cql_alpha",     type=float, default=CQL_ALPHA)
    p.add_argument("--n_candidates",  type=int,   default=NUM_CANDIDATES)
    p.add_argument("--noise_std",     type=float, default=NOISE_STD)


    # Optional PINN evaluation
    p.add_argument("--eval_in_env",   action="store_true", default=False,
                   help="Enable per-step PINN evaluation during training")
    p.add_argument("--pinn_dir",      type=str,
                   default="results/pinn/individual")
    p.add_argument("--csv",           type=str,
                   default=None,
                   help="Optional processed MIMIC CSV for development-only environment evaluation fallback. "
                    "Not required for offline training when init_state_norm.npy is available.")
    p.add_argument(
        "--no_mimic_csv",
        action="store_true",
        help="Do not read MIMIC CSV; require init_state_norm.npy under each patient's PINN dir.",
    )
    p.add_argument("--eval_interval", type=int,   default=EVAL_INTERVAL)
    p.add_argument("--eval_episodes", type=int,   default=5)
    p.add_argument("--k",             type=int,   default=20)
    p.add_argument("--lactate_threshold", type=float, default=8.5)

    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_root = os.path.join(args.save_dir, "models")
    log_root   = os.path.join(args.save_dir, "logs")
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(log_root,   exist_ok=True)

    datasets = discover_datasets(args.dataset_dir)

    target_tags = None
    if args.tags:
        # Preserve existing substring-based behavior for manual --tags.
        datasets = [(tag, path) for tag, path in datasets
                    if any(t in tag for t in args.tags)]
    elif args.patient_ids_file:
        target_tags = tags_from_patient_ids_file(args.patient_ids_file, args.dt_mode)
        datasets = [(tag, path) for tag, path in datasets if tag in target_tags]

        found_tags = {tag for tag, _ in datasets}
        missing_tags = sorted(target_tags - found_tags)
        if missing_tags:
            print(
                f"  [Warning] {len(missing_tags)} requested dataset tags were not found "
                f"under {args.dataset_dir}; first missing tags: {missing_tags[:10]}",
                flush=True,
            )

    print(f"\n{'='*60}", flush=True)
    print(f"  train_cql_multi  —  device: {DEVICE}", flush=True)
    print(f"  dataset_dir : {args.dataset_dir}", flush=True)
    print(f"  save_dir    : {args.save_dir}", flush=True)
    print(f"  patient_ids_file : {args.patient_ids_file}", flush=True)
    print(f"  dt_mode     : {args.dt_mode}", flush=True)
    print(f"  datasets    : {len(datasets)}", flush=True)
    for tag, path in datasets:
        print(f"    {tag}", flush=True)
    print(f"{'='*60}\n", flush=True)

    summary_records = []

    for tag, npz_path in datasets:
        model_dir = os.path.join(model_root, tag)

        # Build optional PINN env (only for per-patient datasets)
        pinn_env = pinn_init_norm = None
        csv_path = None if args.no_mimic_csv else args.csv
        if args.eval_in_env:
            pid = parse_pid_from_tag(tag)
            if pid is not None:
                try:
                    pinn_env, pinn_init_norm = build_pinn_env(
                        args.pinn_dir, csv_path, pid, args)
                    print(f"  [PINN ENV] pid={pid} ready", flush=True)
                except Exception as e:
                    print(f"  [Warning] PINN env for pid={pid} failed: {e}", flush=True)
            else:
                # Population model — skip per-patient PINN eval
                print(f"  [Info] No PINN eval for population dataset: {tag}", flush=True)

        log, eval_log, baseline = train_one(
            npz_path, model_dir, log_root, tag, args,
            pinn_env=pinn_env, pinn_init_norm=pinn_init_norm,
        )

        # Collect summary
        final_eval_r    = eval_log["reward"][-1] if eval_log["reward"] else float("nan")
        final_eval_sofa = eval_log["sofa"][-1]   if eval_log["sofa"]   else float("nan")
        summary_records.append({
            "tag":           tag,
            "baseline_r":    baseline["r_mean"],
            "baseline_sofa": baseline["sofa_mean"],
            "final_eval_r":  final_eval_r,
            "final_eval_sofa": final_eval_sofa,
            "final_q":       float(np.mean(log["q_data_mean"][-100:])),
            "final_cql_pen": float(np.mean(log["cql_penalty"][-100:])),
        })

    # Print overall summary table
    print(f"\n{'='*80}", flush=True)
    print(f"{'Tag':<35} {'BaseR':>7} {'EvalR':>7} {'BaseSofa':>9} {'EvalSofa':>9}",
          flush=True)
    print("-" * 80, flush=True)
    for rec in summary_records:
        print(
            f"{rec['tag']:<35} "
            f"{rec['baseline_r']:>7.3f} "
            f"{rec['final_eval_r']:>7.3f} "
            f"{rec['baseline_sofa']:>9.3f} "
            f"{rec['final_eval_sofa']:>9.3f}",
            flush=True,
        )
    print("=" * 80, flush=True)

    # Save summary CSV
    csv_lines = ["tag,baseline_r,final_eval_r,baseline_sofa,final_eval_sofa,"
                 "final_q,final_cql_pen"]
    for rec in summary_records:
        csv_lines.append(
            f"{rec['tag']},{rec['baseline_r']:.4f},{rec['final_eval_r']:.4f},"
            f"{rec['baseline_sofa']:.4f},{rec['final_eval_sofa']:.4f},"
            f"{rec['final_q']:.4f},{rec['final_cql_pen']:.4f}"
        )
    csv_path = os.path.join(args.save_dir, "training_summary.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\n[Summary CSV] → {csv_path}", flush=True)
    print(f"[Models]      → {model_root}", flush=True)
    print(f"[Logs]        → {log_root}", flush=True)


if __name__ == "__main__":
    main()
