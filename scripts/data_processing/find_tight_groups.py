"""
find_tight_groups.py
=====================
From a PINN pairwise distance matrix, discover multiple ~10-patient groups that are
tight within-group and well separated between groups.

Algorithm:
  1. For each patient, mean distance to the k nearest neighbors = cohesion score.
  2. Prefer patients with low cohesion (dense local neighborhood) as center seeds.
  3. Greedy center selection:
       - Pick the unassigned patient with minimum cohesion as the next center.
       - Assign its k-1 nearest unassigned neighbors to the same group.
       - Require each new center to be at least min_center_dist from existing centers.
  4. Keep only groups whose mean within-group distance satisfies within_thresh.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir",    default="results_tight_groups")
parser.add_argument("--in-dir",     default="results_pinn_pairwise_dist")
parser.add_argument("--group-size", type=int,   default=10)
parser.add_argument("--max-groups", type=int,   default=30)
# Comma-separated search grids for within_thresh and min_center_dist
parser.add_argument("--within-list",  default="0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0")
parser.add_argument("--center-list",  default="0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0")
args = parser.parse_args()

ROOT    = Path(__file__).resolve().parent
IN_DIR  = ROOT / args.in_dir
OUT_DIR = ROOT / args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_SIZE      = args.group_size
MAX_GROUPS      = args.max_groups
WITHIN_THRESHES = [float(x) for x in args.within_list.split(",")]
CENTER_DISTS    = [float(x) for x in args.center_list.split(",")]


def find_tight_groups(D, pids, group_size=10, within_thresh=2.0,
                      min_center_dist=4.0, max_groups=30):
    N = len(pids)
    # 1) Mean distance to k nearest neighbors (cohesion score); diagonal is zero.
    sorted_d = np.sort(D, axis=1)           # (N, N) ascending
    cohesion = sorted_d[:, 1:group_size+1].mean(axis=1)  # mean over k neighbors

    assigned  = np.zeros(N, dtype=bool)
    centers   = []   # center indices
    groups    = []   # list of (center_idx, member_idx_list, within_mean)

    # 2) Greedy center selection
    for _ in range(max_groups):
        # Unassigned patients only; next center has minimum cohesion among them
        cand_mask = ~assigned
        if cand_mask.sum() < group_size:
            break

        # Also require distance >= min_center_dist from every existing center
        if centers:
            dist_to_centers = D[:, centers].min(axis=1)  # (N,)
            cand_mask &= (dist_to_centers >= min_center_dist)

        if cand_mask.sum() == 0:
            break

        # Lowest cohesion among remaining candidates
        cand_idx  = np.where(cand_mask)[0]
        best_c    = cand_idx[np.argmin(cohesion[cand_idx])]

        # Take best_c plus group_size-1 nearest unassigned neighbors
        neighbor_order = np.argsort(D[best_c])  # ascending distance from best_c
        members = [best_c]
        for nb in neighbor_order:
            if nb == best_c:
                continue
            if not assigned[nb]:
                members.append(nb)
            if len(members) == group_size:
                break

        if len(members) < group_size:
            break

        # within mean
        member_arr = np.array(members)
        sub_d = D[np.ix_(member_arr, member_arr)]
        wi = sub_d[np.triu_indices(len(members), k=1)].mean()

        if wi > within_thresh:
            # Stop if even the densest feasible group violates within_thresh
            break

        assigned[member_arr] = True
        centers.append(best_c)
        groups.append({
            "group_id":    len(groups),
            "center_idx":  best_c,
            "member_idxs": member_arr,
            "within_mean": round(wi, 3),
            "n":           len(members),
        })

    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Load inputs
# ═══════════════════════════════════════════════════════════════════════════════
D    = np.load(IN_DIR / "distance_matrix.npy")
D    = (D + D.T) / 2; np.fill_diagonal(D, 0.); D = np.clip(D, 0., None)
pids = np.load(IN_DIR / "valid_pids.npy")
emb  = np.load(IN_DIR / "mds_embedding.npy")
N    = len(pids)
print(f"Patients: {N}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Grid search over within_thresh and min_center_dist
# ═══════════════════════════════════════════════════════════════════════════════
results_summary = []
best_groups, best_cfg = None, None

for within_t in WITHIN_THRESHES:
    for center_d in CENTER_DISTS:
        groups = find_tight_groups(
            D, pids,
            group_size      = GROUP_SIZE,
            within_thresh   = within_t,
            min_center_dist = center_d,
            max_groups      = MAX_GROUPS,
        )
        if not groups:
            continue
        n_groups    = len(groups)
        withing_avg = np.mean([g["within_mean"] for g in groups])

        # Between-group distances (center pairs)
        if n_groups >= 2:
            c_idxs  = [g["center_idx"] for g in groups]
            bt_dists = D[np.ix_(c_idxs, c_idxs)]
            bt_avg   = bt_dists[np.triu_indices(n_groups, k=1)].mean()
        else:
            bt_avg = 0.

        ratio = bt_avg / withing_avg if withing_avg > 0 else 0.
        print(f"  within_t={within_t}  center_d={center_d}  "
              f"→ {n_groups} groups  within={withing_avg:.2f}  "
              f"between={bt_avg:.2f}  ratio={ratio:.2f}", flush=True)
        results_summary.append({
            "within_t": within_t, "center_d": center_d,
            "n_groups": n_groups, "within_avg": withing_avg,
            "between_avg": bt_avg, "ratio": ratio,
        })
        cur_cfg = results_summary[-1]
        if best_groups is None:
            best_groups = groups
            best_cfg = cur_cfg
        else:
            cur_full = cur_cfg["n_groups"] == MAX_GROUPS
            best_full = best_cfg["n_groups"] == MAX_GROUPS

            if cur_full and not best_full:
                best_groups = groups
                best_cfg = cur_cfg
            elif cur_full and best_full and cur_cfg["within_avg"] < best_cfg["within_avg"]:
                best_groups = groups
                best_cfg = cur_cfg
            elif not cur_full and not best_full:
                if (
                    cur_cfg["n_groups"] > best_cfg["n_groups"]
                    or (
                        cur_cfg["n_groups"] == best_cfg["n_groups"]
                        and cur_cfg["within_avg"] < best_cfg["within_avg"]
                    )
                ):
                    best_groups = groups
                    best_cfg = cur_cfg

print(f"\nSelected config: {best_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Write outputs
# ═══════════════════════════════════════════════════════════════════════════════
ng = len(best_groups)
print(f"\n=== {ng} groups ===", flush=True)
rows = []
for g in best_groups:
    center_pid = int(pids[g["center_idx"]])
    member_pids = pids[g["member_idxs"]].tolist()
    print(f"  G{g['group_id']:2d}  center={center_pid}  within={g['within_mean']:.3f}"
          f"  members={member_pids}", flush=True)
    for member_idx, pid in zip(g["member_idxs"], member_pids):
        rows.append({"group": g["group_id"], "pid": int(pid),
                     "role": "center" if int(pid) == center_pid else "member",
                     "within_mean": g["within_mean"],
                     "distance_to_medoid": float(D[g["center_idx"], member_idx])})

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / "tight_group_assignments.csv", index=False)
print(f"\n[Saved] tight_group_assignments.csv", flush=True)

# cluster_assignments.csv for downstream scripts (e.g. eval_pinn_dist_cluster.py)
df_cl = df_out.rename(columns={"group": "cluster"})[
    ["cluster", "pid", "role", "within_mean", "distance_to_medoid"]
]
df_cl.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)
print(f"[Saved] cluster_assignments.csv", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════
colors = cm.tab20(np.linspace(0, 1, max(ng, 1)))

# 1) MDS scatter
fig, ax = plt.subplots(figsize=(10, 8))
# Unassigned patients in light gray
assigned_all = np.concatenate([g["member_idxs"] for g in best_groups])
unassigned   = np.setdiff1d(np.arange(N), assigned_all)
ax.scatter(emb[unassigned, 0], emb[unassigned, 1],
           s=8, alpha=0.25, color="lightgray", label="Unassigned")

for g in best_groups:
    idx = g["member_idxs"]
    ci  = g["center_idx"]
    ax.scatter(emb[idx, 0], emb[idx, 1], s=30, alpha=0.8,
               color=colors[g["group_id"]],
               label=f"G{g['group_id']} (w={g['within_mean']:.2f})")
    ax.scatter(emb[ci, 0], emb[ci, 1], s=100, marker="*",
               color=colors[g["group_id"]], edgecolors="black", lw=0.8, zorder=10)

# Lines between group centers
c_idxs = [g["center_idx"] for g in best_groups]
for i in range(len(c_idxs)):
    for j in range(i+1, len(c_idxs)):
        x0, y0 = emb[c_idxs[i]]
        x1, y1 = emb[c_idxs[j]]
        ax.plot([x0, x1], [y0, y1], "k-", lw=0.3, alpha=0.2)

ax.set_title(f"{ng} tight groups  "
             f"within≤{best_cfg['within_t']}  center_dist≥{best_cfg['center_d']}\n"
             f"within_avg={best_cfg['within_avg']:.2f}  "
             f"between_avg={best_cfg['between_avg']:.2f}  "
             f"ratio={best_cfg['ratio']:.2f}×",
             fontsize=9)
ax.set_xlabel("MDS dim 1"); ax.set_ylabel("MDS dim 2")
ax.legend(fontsize=6, ncol=3, loc="best"); ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(OUT_DIR / "tight_groups_mds.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[Saved] tight_groups_mds.png", flush=True)

# 2) Within vs between distance histograms
w_all, b_all = [], []
for g in best_groups:
    idx = g["member_idxs"]
    sub = D[np.ix_(idx, idx)]
    w_all.extend(sub[np.triu_indices(len(idx), k=1)].tolist())
for i, gi in enumerate(best_groups):
    for gj in best_groups[i+1:]:
        b_all.extend(D[np.ix_(gi["member_idxs"], gj["member_idxs"])].ravel().tolist())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.hist(w_all, bins=40, alpha=0.7, color="steelblue",
        label=f"Within  μ={np.mean(w_all):.2f}", density=True)
ax.hist(b_all[:len(w_all)*5], bins=40, alpha=0.7, color="salmon",
        label=f"Between μ={np.mean(b_all):.2f}", density=True)
ax.set_xlabel("Distance"); ax.set_ylabel("Density")
ax.set_title("Within vs Between distance distribution")
ax.legend(); ax.grid(alpha=0.3)

# 3) Per-group within-distance bar chart
ax = axes[1]
within_vals = [g["within_mean"] for g in best_groups]
ax.barh(range(ng), within_vals, color=[colors[i] for i in range(ng)])
ax.axvline(best_cfg["within_t"], color="red", ls="--", label=f"thresh={best_cfg['within_t']}")
ax.set_yticks(range(ng)); ax.set_yticklabels([f"G{g['group_id']}" for g in best_groups])
ax.set_xlabel("Within-group mean distance")
ax.set_title("Per-group within distance")
ax.legend(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "tight_groups_stats.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[Saved] tight_groups_stats.png", flush=True)

# 4) Center-to-center distance matrix heatmap
if ng >= 2:
    c_idxs = [g["center_idx"] for g in best_groups]
    D_centers = D[np.ix_(c_idxs, c_idxs)]
    fig, ax = plt.subplots(figsize=(max(5, ng//2), max(4, ng//2)))
    im = ax.imshow(D_centers, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="center-to-center distance")
    ax.set_xticks(range(ng)); ax.set_xticklabels([f"G{g['group_id']}" for g in best_groups], fontsize=7)
    ax.set_yticks(range(ng)); ax.set_yticklabels([f"G{g['group_id']}" for g in best_groups], fontsize=7)
    for i in range(ng):
        for j in range(ng):
            ax.text(j, i, f"{D_centers[i,j]:.1f}", ha="center", va="center", fontsize=6)
    ax.set_title("Center-to-center distances (green=far=good)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "group_center_dist_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] group_center_dist_matrix.png", flush=True)

print(f"\nAll outputs: {OUT_DIR}", flush=True)
