#!/usr/bin/env python3
"""collect_cluster_test_data.py

Step 1 of the cluster offline RL pipeline.

Collect offline rollout data for each cluster's test patient using the
corresponding individual (Ind) online policy on the patient's own PINN.

Reads test patients and cluster assignments from --cluster_map.

Output:
  <out_dir>/datasets/ind_<pid>_fixdt_dataset.npz   (1 000 transitions/patient)
"""
from __future__ import annotations
import argparse, csv, sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [PROJECT_ROOT, Path(__file__).resolve().parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from models import PINN
from rl import ICUEnvironment, LagrangianTRPO, SMDP_STATE_DIM, OPTION_ACTION_DIM
from data.config import action_dim, state_dim, device
from pinn_bundle import load_init_and_scales_for_patient
from collect_offline_data import _collect_policy, TOTAL_TIME_H


def load_test_patients(cluster_map_csv: Path) -> dict[int, int]:
    """Return {patient_id: cluster_id} for rows with role/split == 'test'."""
    result = {}
    with open(cluster_map_csv) as f:
        for row in csv.DictReader(f):
            if row.get("role", row.get("split", "test")).strip() == "test":
                result[int(row["patient_id"])] = int(row["cluster_id"])
    return result


def _make_patient_env(pinn_path, mean_pat, std_pat, asc_pat,
                      state_min, state_max, K: int) -> ICUEnvironment:
    fixed_dt = TOTAL_TIME_H / K
    pinn = PINN(state_dim, action_dim).to(device)
    pinn.load_state_dict(torch.load(str(pinn_path), map_location=device))
    pinn.eval()
    return ICUEnvironment(
        pinn_model      = pinn,
        mean_np         = mean_pat,
        std_np          = std_pat,
        action_min_norm = np.zeros(action_dim, dtype=np.float32),
        action_max_norm = np.ones(action_dim,  dtype=np.float32),
        state_min       = state_min,
        state_max       = state_max,
        action_scale_np = asc_pat,
        max_steps       = K,
        dt_min          = 0.5,
        dt_max          = 36.0,
        total_time_h    = TOTAL_TIME_H,
        use_dt          = False,
        fixed_dt        = fixed_dt,
        use_lac_penalty = True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_map", type=Path,
                   default=PROJECT_ROOT / "cohort_splits" / "cohort_1" / "cluster_map.csv")
    p.add_argument("--out_dir",  type=Path,
                   default=PROJECT_ROOT / "results" / "offline_cluster" / "cohort_1")
    p.add_argument("--pinn_dir", type=Path,
                   default=PROJECT_ROOT / "results" / "pinn" / "individual")
    p.add_argument("--ind_root", type=Path,
                   default=PROJECT_ROOT / "results" / "online" / "individual")
    p.add_argument("--algo",     default="lagrangian_trpo")
    p.add_argument("--tag",      default="fixdt")
    p.add_argument("--K",        type=int, default=20)
    p.add_argument("--n_eval",   type=int, default=50)
    p.add_argument("--noise",    type=float, default=0.05)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--skip_done", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    test_patients = load_test_patients(args.cluster_map)
    if not test_patients:
        raise ValueError(f"No test patients found in {args.cluster_map}")

    ds_dir = args.out_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: Collecting Ind rollouts  algo={args.algo}  n_eval={args.n_eval}")
    print(f"  test patients ({len(test_patients)}): {sorted(test_patients.keys())}")

    for idx, (pid, cid) in enumerate(sorted(test_patients.items(), key=lambda x: x[1])):
        out_path = ds_dir / f"ind_{pid}_{args.tag}_dataset.npz"
        if args.skip_done and out_path.exists():
            print(f"  [SKIP] pid={pid}")
            continue
        print(f"\n  [{idx+1:02d}/{len(test_patients)}] pid={pid}  cluster={cid}", flush=True)

        try:
            init_norm, pinn_path, mean_pat, std_pat, asc_pat, state_min, state_max = \
                load_init_and_scales_for_patient(args.pinn_dir, pid, None)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [SKIP] {e}")
            continue

        env = _make_patient_env(pinn_path, mean_pat, std_pat, asc_pat,
                                state_min, state_max, args.K)
        init_norm = np.asarray(init_norm, dtype=np.float32)

        pt_dir = args.ind_root / f"patient_{pid}" / args.algo / f"K{args.K}" / args.tag
        pt = next((pt_dir / n for n in ["policy.pt", "actor.pt"] if (pt_dir / n).exists()), None)
        if pt is None:
            print(f"  [MISSING policy] {pt_dir}")
            continue

        sd = torch.load(str(pt), map_location=device, weights_only=False)
        actor_sd = sd["actor"] if isinstance(sd, dict) and "actor" in sd else sd
        hidden_size = actor_sd["net.0.weight"].shape[0]
        agent = LagrangianTRPO(state_dim=SMDP_STATE_DIM, action_dim=OPTION_ACTION_DIM,
                               hidden=hidden_size)
        agent.actor.load_state_dict(actor_sd)

        rng = np.random.default_rng(args.seed + idx)
        *_, dataset = _collect_policy(agent, env, init_norm, args.n_eval,
                                      action_noise_std=args.noise, rng=rng)

        n = dataset["transitions/r"].shape[0]
        np.savez_compressed(out_path, **dataset,
                            **{"info/patient_id":    np.array([pid]),
                               "info/cluster_id":    np.array([cid]),
                               "info/policy_tag":    np.array([f"ind_{args.algo}_{args.tag}"]),
                               "info/n_transitions": np.array([n]),
                               "info/n_eval":        np.array([args.n_eval])})
        print(f"  [saved] {out_path.name}  ({n} transitions)", flush=True)

    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()
