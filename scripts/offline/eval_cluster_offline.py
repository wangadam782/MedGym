#!/usr/bin/env python3
"""eval_cluster_offline.py

Step 5 of the cluster offline RL pipeline.

Evaluates Ind / Cluster-pooled / Population offline policies on each
cluster's test patient using the patient's own PINN simulator.

Reads test patients and cluster assignments from --cluster_map.

Output:
  <eval_root>/<scope>/<method>/summary/per_patient_metrics.csv
  <eval_root>/<scope>/<method>/datasets/*.npz
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
from rl import ICUEnvironment
from data.config import action_dim, state_dim, device
from pinn_bundle import load_init_and_scales_for_patient
import eval_offline as offline_eval
from collect_offline_data import TOTAL_TIME_H

METHODS = ("dqn", "cql", "gcql")
SCOPES  = ("ind", "clu", "pop")


def load_test_patients(cluster_map_csv: Path) -> dict[int, int]:
    result = {}
    with open(cluster_map_csv) as f:
        for row in csv.DictReader(f):
            if row["role"].strip() == "test":
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


def _dataset_final_sofa(data: dict) -> float:
    sofa  = np.asarray(data["meta/sofa"],  dtype=np.float32)
    ep_id = np.asarray(data["meta/ep_id"], dtype=np.int32)
    vals  = [float(sofa[np.where(ep_id == ep)[0][-1]])
             for ep in np.unique(ep_id)]
    return float(np.mean(vals)) if vals else float("nan")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_map", type=Path,
                   default=PROJECT_ROOT / "cohort_splits" / "cohort_1" / "cluster_map.csv")
    p.add_argument("--plan_a_root", type=Path,
                   default=PROJECT_ROOT / "results" / "offline_cluster" / "cohort_1",
                   help="Root for trained-from-scratch runs; policy_root defaults to plan_a_root/policies.")
    p.add_argument("--policy_root", type=Path, default=None,
                   help="Override policy directory directly (e.g. checkpoints-cohort/cohort_1/offline). "
                        "Defaults to --plan_a_root/policies.")
    p.add_argument("--eval_root", type=Path, default=None,
                   help="Override eval output directory. Defaults to --plan_a_root/eval.")
    p.add_argument("--pinn_dir", type=Path,
                   default=PROJECT_ROOT / "results" / "pinn" / "individual")
    p.add_argument("--tag",      default="fixdt")
    p.add_argument("--K",        type=int, default=20)
    p.add_argument("--n_eval",   type=int, default=50)
    p.add_argument("--noise",    type=float, default=0.05)
    p.add_argument("--hidden",   type=int, default=256)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--scopes",   nargs="+", default=list(SCOPES), choices=SCOPES)
    p.add_argument("--methods",  nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--clu_by_pid", action="store_true", default=False,
                   help="Use patient-ID-based clu model tags (clu_{pid}_fixdt) instead of "
                        "cluster-number-based (cluster{cid}_fixdt). Use for checkpoints-remaining15.")
    p.add_argument("--skip_done", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args          = parse_args()
    test_patients = load_test_patients(args.cluster_map)
    if not test_patients:
        raise ValueError(f"No test patients found in {args.cluster_map}")

    policy_root = args.policy_root if args.policy_root else args.plan_a_root / "policies"
    eval_root   = args.eval_root   if args.eval_root   else args.plan_a_root / "eval"
    lac_thr     = getattr(offline_eval, "LAC_DANGER_THR", 4.0)

    print(f"Offline cluster eval  n_eval={args.n_eval}")
    print(f"  scopes={args.scopes}  methods={args.methods}")
    print(f"  patients ({len(test_patients)}): {sorted(test_patients.keys())}\n")

    for method in args.methods:
        print(f"\n=== method={method} ===")
        scope_rows: dict[str, list] = {s: [] for s in SCOPES}

        for idx, (pid, cid) in enumerate(sorted(test_patients.items(), key=lambda x: x[1])):
            try:
                init_norm, pinn_path, mean_pat, std_pat, asc_pat, state_min, state_max = \
                    load_init_and_scales_for_patient(args.pinn_dir, pid, None)
            except (FileNotFoundError, ValueError) as e:
                print(f"  [SKIP pid={pid}] {e}")
                continue

            env       = _make_patient_env(pinn_path, mean_pat, std_pat, asc_pat,
                                          state_min, state_max, args.K)
            init_norm = np.asarray(init_norm, dtype=np.float32)

            for scope in args.scopes:
                if scope == "ind":
                    model_tag = f"ind_{pid}_{args.tag}"
                elif scope == "clu":
                    # cohort: cluster{cid}_fixdt  /  remaining15: clu_{pid}_fixdt
                    model_tag = f"clu_{pid}_{args.tag}" if args.clu_by_pid \
                                else f"cluster{cid}_{args.tag}"
                else:
                    model_tag = f"pop_{args.tag}"

                model_path = policy_root / scope / method / "models" / model_tag / "actor_final.pt"
                if not model_path.exists():
                    print(f"  [MISSING] {model_path}")
                    continue

                out_dir  = eval_root / scope / method
                stem     = f"{model_tag}_pid{pid}" if scope != "ind" else model_tag
                npz_path = out_dir / "datasets" / f"{stem}_{method}_{args.tag}_eval.npz"

                if args.skip_done and npz_path.exists():
                    print(f"  [SKIP] {npz_path.name}")
                    with np.load(npz_path, allow_pickle=False) as d:
                        metrics = {
                            "mean_sofa_all": float(np.mean(d["meta/sofa"])),
                            "final_sofa":    _dataset_final_sofa({k: d[k] for k in d.files}),
                            "safety_rate":   float(np.mean(d["meta/lactate"] < lac_thr)),
                            "mean_lac":      float(np.mean(d["meta/lactate"])),
                        }
                else:
                    actor  = offline_eval._load_actor(str(model_path), hidden=args.hidden)
                    rng    = np.random.default_rng(args.seed + idx * 100 + SCOPES.index(scope))
                    *_, metrics_raw, dataset = offline_eval._collect_policy(
                        actor, env, init_norm, args.n_eval,
                        action_noise_std=args.noise, rng=rng
                    )
                    metrics = dict(metrics_raw)
                    metrics["final_sofa"]    = _dataset_final_sofa(dataset)
                    metrics["mean_sofa_all"] = float(np.mean(dataset["meta/sofa"]))
                    metrics["safety_rate"]   = float(np.mean(dataset["meta/lactate"] < lac_thr))

                    npz_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(npz_path, **dataset,
                                        **{"info/method":       np.array([method]),
                                           "info/scope":        np.array([scope]),
                                           "info/model_tag":    np.array([model_tag]),
                                           "info/patient_id":   np.array([pid]),
                                           "info/cluster_id":   np.array([cid]),
                                           "info/n_transitions":np.array([dataset["transitions/r"].shape[0]]),
                                           "info/n_eval":       np.array([args.n_eval])})

                scope_rows[scope].append({
                    "method": method, "scope": scope, "tag": args.tag,
                    "patient_id": pid, "cluster_id": cid, "model_tag": model_tag,
                    "mean_sofa_all": metrics["mean_sofa_all"],
                    "final_sofa":    metrics["final_sofa"],
                    "safety_rate":   metrics["safety_rate"],
                    "mean_lac":      metrics.get("mean_lac", float("nan")),
                    "rollout_npz":   str(npz_path),
                })
                print(f"  [{method}|{scope}] pid={pid} clu={cid} "
                      f"final_sofa={metrics['final_sofa']:.3f} "
                      f"safe={metrics['safety_rate']*100:.1f}%", flush=True)

        fields = ["method", "scope", "tag", "patient_id", "cluster_id", "model_tag",
                  "mean_sofa_all", "final_sofa", "safety_rate", "mean_lac", "rollout_npz"]
        for scope in args.scopes:
            rows = scope_rows[scope]
            if not rows:
                continue
            csv_path = eval_root / scope / method / "summary" / "per_patient_metrics.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in fields} for r in rows)
            print(f"  [saved] {csv_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
