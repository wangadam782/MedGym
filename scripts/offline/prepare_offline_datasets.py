#!/usr/bin/env python3
"""prepare_offline_datasets.py

Step 2/3 of the cluster offline RL pipeline.

Step 2 — Copy cluster-pooled behavior datasets:
  Renames fm_clu_cluster{k}_fixdt_dataset.npz  →  cluster{k}_fixdt_dataset.npz
  (these were collected by the cluster-pooled online RL behavior policy)

Step 3 — Build population dataset from per-training-patient datasets:
  Samples n_per_patient transitions from each training patient's dataset
  and merges them into a single pop_{tag}_dataset.npz
  (100 training patients × 10 transitions = 1000 transitions by default)

Usage:
    python scripts/offline/prepare_offline_datasets.py \\
        --out_dir  results/offline_cluster/cohort_1 \\
        --src_dir  results/offline_cluster/cohort_1/datasets/raw \\
        --n_clusters 10

Expected source naming in --src_dir:
    fm_clu_cluster{1..n_clusters}_fixdt_dataset.npz
    clu_per_training_patient/fm_clu_tp*_fixdt_dataset.npz
"""
from __future__ import annotations
import argparse, shutil
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",  type=Path,
                   default=PROJECT_ROOT / "results" / "offline_cluster" / "cohort_1")
    p.add_argument("--src_dir",  type=Path,
                   default=PROJECT_ROOT / "results" / "offline_cluster" / "cohort_1" / "datasets" / "raw",
                   help="Directory containing fm_clu_cluster*.npz and clu_per_training_patient/")
    p.add_argument("--tag",         default="fixdt")
    p.add_argument("--n_clusters",  type=int, default=10)
    p.add_argument("--n_per_patient", type=int, default=10,
                   help="Transitions to sample from each training patient for pop dataset")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--skip_done",   action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    ds_dir = args.out_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 2: Copy cluster datasets ─────────────────────────────────────────
    print("=== Step 2: Preparing cluster datasets ===")
    for k in range(1, args.n_clusters + 1):
        src = args.src_dir / f"fm_clu_cluster{k}_{args.tag}_dataset.npz"
        dst = ds_dir / f"cluster{k}_{args.tag}_dataset.npz"
        if args.skip_done and dst.exists():
            print(f"  [SKIP] cluster{k}")
            continue
        if not src.exists():
            print(f"  [MISSING] {src}")
            continue
        d = np.load(src, allow_pickle=True)
        n = d["transitions/r"].shape[0]
        shutil.copy2(src, dst)
        print(f"  [copied] cluster{k}: {n} transitions")

    # ── Step 3: Build population dataset ──────────────────────────────────────
    print("\n=== Step 3: Building population dataset ===")
    dst_pop = ds_dir / f"pop_{args.tag}_dataset.npz"
    if args.skip_done and dst_pop.exists():
            print("  [SKIP] pop")
    else:
        per_patient_dir = args.src_dir / "clu_per_training_patient"
        per_patient_files = sorted(per_patient_dir.glob(f"fm_clu_tp*_{args.tag}_dataset.npz"))
        if not per_patient_files:
            print(f"  [SKIP] No per-patient files found in {per_patient_dir}")
        else:
            rng = np.random.default_rng(args.seed)
            fields = ["transitions/s", "transitions/a", "transitions/r",
                      "transitions/ns", "transitions/done", "meta/sofa", "meta/lactate"]
            parts = {f: [] for f in fields}
            ep_ids_all, ep_lens, ep_rews, source_pids = [], [], [], []
            ep_offset = 0

            for f in per_patient_files:
                d = np.load(f, allow_pickle=True)
                n_total = d["transitions/r"].shape[0]
                n_take = min(args.n_per_patient, n_total)
                idx = rng.choice(n_total, size=n_take, replace=False)
                idx.sort()
                for field in fields:
                    parts[field].append(d[field][idx])
                ep_ids_all.append(np.full(n_take, ep_offset, dtype=np.int32))
                ep_lens.append(n_take)
                ep_rews.append(float(d["transitions/r"][idx].sum()))
                if "info/training_patient_id" in d.files:
                    source_pids.append(int(d["info/training_patient_id"][0]))
                ep_offset += 1

            merged = {f: np.concatenate(parts[f], axis=0) for f in fields}
            total = int(merged["transitions/r"].shape[0])
            merged["meta/ep_id"]              = np.concatenate(ep_ids_all)
            merged["meta/ep_length"]          = np.array(ep_lens, dtype=np.int32)
            merged["meta/ep_reward_sum"]      = np.array(ep_rews, dtype=np.float32)
            merged["info/n_transitions"]      = np.array([total])
            merged["info/n_source_patients"]  = np.array([len(per_patient_files)])
            merged["info/n_per_patient"]      = np.array([args.n_per_patient])
            if source_pids:
                merged["info/source_patient_ids"] = np.array(source_pids)
            merged["info/policy_tag"]         = np.array([f"pop_ind_{args.tag}"])

            np.savez_compressed(dst_pop, **merged)
                print(f"  [saved] pop: {total} transitions "
                  f"({len(per_patient_files)} patients × {args.n_per_patient})")

    print("\n=== Dataset summary ===")
    for f in sorted(ds_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        n = d["transitions/r"].shape[0]
        print(f"  {f.name}: {n} transitions")


if __name__ == "__main__":
    main()
