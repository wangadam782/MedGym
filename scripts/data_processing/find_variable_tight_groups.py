#!/usr/bin/env python3
"""Find variable-size tight/far patient groups from a PINN distance matrix.

This script is meant for exploratory clustering when the cohort should not be
forced into a fixed K or fixed cluster size.  It uses complete-linkage
agglomeration for each within-cluster threshold, so initial clusters have
bounded pairwise diameter.  Small clusters are left unassigned; clusters whose
medoids are too close can either be kept, rejected, or merged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cluster_diameter(distance: np.ndarray, cluster: list[int]) -> float:
    if len(cluster) < 2:
        return 0.0
    sub = distance[np.ix_(cluster, cluster)]
    return float(sub[np.triu_indices(len(cluster), k=1)].max())


def cluster_mean_pairwise(distance: np.ndarray, cluster: list[int]) -> float:
    if len(cluster) < 2:
        return 0.0
    sub = distance[np.ix_(cluster, cluster)]
    return float(sub[np.triu_indices(len(cluster), k=1)].mean())


def medoid_index(distance: np.ndarray, cluster: list[int]) -> int:
    sub = distance[np.ix_(cluster, cluster)]
    return int(cluster[int(np.argmin(sub.sum(axis=1)))])


def complete_link_distance(distance: np.ndarray, left: list[int], right: list[int]) -> float:
    return float(distance[np.ix_(left, right)].max())


def complete_linkage_clusters(distance: np.ndarray, threshold: float) -> list[list[int]]:
    """Return a partition whose cluster diameters are <= threshold."""
    clusters: list[list[int]] = [[i] for i in range(distance.shape[0])]
    while True:
        best_pair: tuple[int, int] | None = None
        best_dist = math.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = complete_link_distance(distance, clusters[i], clusters[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        if best_pair is None or best_dist > threshold:
            break
        i, j = best_pair
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]
    return clusters


def silhouette_assigned(distance: np.ndarray, clusters: list[list[int]]) -> float:
    if len(clusters) < 2:
        return 0.0
    labels: dict[int, int] = {}
    for cid, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cid
    values = []
    for idx, own_cid in labels.items():
        own = [x for x in clusters[own_cid] if x != idx]
        a = float(distance[idx, own].mean()) if own else 0.0
        b = min(
            float(distance[idx, other].mean())
            for cid, other in enumerate(clusters)
            if cid != own_cid
        )
        denom = max(a, b)
        values.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(values)) if values else 0.0


def inter_cluster_stats(distance: np.ndarray, clusters: list[list[int]], medoids: list[int]) -> dict[str, float]:
    if len(clusters) < 2:
        return {
            "min_medoid_distance": 0.0,
            "mean_medoid_distance": 0.0,
            "min_inter_patient_distance": 0.0,
            "mean_nearest_inter_patient_distance": 0.0,
        }
    medoid_vals = []
    nearest_vals = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            medoid_vals.append(float(distance[medoids[i], medoids[j]]))
            nearest_vals.append(float(distance[np.ix_(clusters[i], clusters[j])].min()))
    return {
        "min_medoid_distance": min(medoid_vals),
        "mean_medoid_distance": float(np.mean(medoid_vals)),
        "min_inter_patient_distance": min(nearest_vals),
        "mean_nearest_inter_patient_distance": float(np.mean(nearest_vals)),
    }


def select_far_clusters(
    distance: np.ndarray,
    clusters: list[list[int]],
    *,
    min_cluster_size: int,
    min_center_dist: float,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    big_enough = [c for c in clusters if len(c) >= min_cluster_size]
    too_small = [c for c in clusters if len(c) < min_cluster_size]
    candidates = sorted(
        big_enough,
        key=lambda c: (
            cluster_mean_pairwise(distance, c),
            cluster_diameter(distance, c),
            -len(c),
            int(medoid_index(distance, c)),
        ),
    )

    accepted: list[list[int]] = []
    rejected_close: list[list[int]] = []
    accepted_medoids: list[int] = []
    for cluster in candidates:
        m = medoid_index(distance, cluster)
        if all(float(distance[m, old_m]) >= min_center_dist for old_m in accepted_medoids):
            accepted.append(cluster)
            accepted_medoids.append(m)
        else:
            rejected_close.append(cluster)
    accepted = sorted(accepted, key=lambda c: int(medoid_index(distance, c)))
    return accepted, too_small, rejected_close


def merge_close_clusters(
    distance: np.ndarray,
    clusters: list[list[int]],
    *,
    min_cluster_size: int,
    min_center_dist: float,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Merge clusters whose medoids are closer than the requested separation."""
    active = [sorted(c) for c in clusters if len(c) >= min_cluster_size]
    too_small = [c for c in clusters if len(c) < min_cluster_size]
    merged_sources: list[list[int]] = []
    if min_center_dist <= 0 or len(active) < 2:
        return sorted(active, key=lambda c: int(medoid_index(distance, c))), too_small, merged_sources

    while len(active) >= 2:
        medoids = [medoid_index(distance, c) for c in active]
        best_pair: tuple[int, int] | None = None
        best_dist = math.inf
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                d = float(distance[medoids[i], medoids[j]])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        if best_pair is None or best_dist >= min_center_dist:
            break
        i, j = best_pair
        active[i] = sorted(active[i] + active[j])
        merged_sources.append(active[j])
        del active[j]

    return sorted(active, key=lambda c: int(medoid_index(distance, c))), too_small, merged_sources


def summarize_config(
    distance: np.ndarray,
    clusters: list[list[int]],
    *,
    n_patients: int,
    within_t: float,
    center_d: float,
    median_pairwise_distance: float,
) -> dict[str, object]:
    sizes = [len(c) for c in clusters]
    assigned = int(sum(sizes))
    medoids = [medoid_index(distance, c) for c in clusters]
    mean_intra_vals = [cluster_mean_pairwise(distance, c) for c in clusters if len(c) > 1]
    diameters = [cluster_diameter(distance, c) for c in clusters]
    mean_intra = float(np.mean(mean_intra_vals)) if mean_intra_vals else 0.0
    max_intra = float(max(diameters)) if diameters else 0.0
    inter = inter_cluster_stats(distance, clusters, medoids)
    mean_medoid = float(inter["mean_medoid_distance"])
    min_medoid = float(inter["min_medoid_distance"])
    ratio = mean_medoid / mean_intra if mean_intra > 0 and len(clusters) > 1 else 0.0
    min_ratio = min_medoid / max_intra if max_intra > 0 and len(clusters) > 1 else 0.0
    silhouette = silhouette_assigned(distance, clusters)

    assigned_frac = assigned / n_patients if n_patients else 0.0
    max_possible = max(1, n_patients // 3)
    n_norm = min(len(clusters) / max_possible, 1.0)
    tight_norm = 1.0 / (1.0 + mean_intra / max(median_pairwise_distance, 1e-8))
    sep_norm = min(ratio / 3.0, 1.5)
    score = 1.5 * assigned_frac + 0.75 * n_norm + 1.5 * tight_norm + 2.0 * sep_norm + silhouette

    return {
        "within_t": within_t,
        "center_d": center_d,
        "n_clusters": len(clusters),
        "assigned_patients": assigned,
        "assigned_frac": assigned_frac,
        "unassigned_patients": n_patients - assigned,
        "cluster_sizes": ";".join(str(s) for s in sizes),
        "min_cluster_size": min(sizes) if sizes else 0,
        "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
        "max_cluster_size": max(sizes) if sizes else 0,
        "mean_intra_distance": mean_intra,
        "max_intra_distance": max_intra,
        "min_medoid_distance": min_medoid,
        "mean_medoid_distance": mean_medoid,
        "min_inter_patient_distance": float(inter["min_inter_patient_distance"]),
        "mean_nearest_inter_patient_distance": float(inter["mean_nearest_inter_patient_distance"]),
        "mean_medoid_over_mean_intra": ratio,
        "min_medoid_over_max_intra": min_ratio,
        "silhouette": silhouette,
        "score": score,
    }


def choose_best(rows: list[dict[str, object]], min_assigned_frac: float) -> dict[str, object]:
    feasible = [r for r in rows if float(r["assigned_frac"]) >= min_assigned_frac and int(r["n_clusters"]) >= 2]
    pool = feasible or [r for r in rows if int(r["n_clusters"]) >= 2] or rows
    return max(
        pool,
        key=lambda r: (
            float(r["score"]),
            int(r["assigned_patients"]),
            int(r["n_clusters"]),
            -float(r["mean_intra_distance"]),
            float(r["mean_medoid_distance"]),
        ),
    )


def plot_clusters(
    out_dir: Path,
    distance: np.ndarray,
    emb: np.ndarray,
    clusters: list[list[int]],
    unassigned: list[int],
) -> None:
    colors = cm.tab20(np.linspace(0, 1, max(len(clusters), 1)))
    fig, ax = plt.subplots(figsize=(10, 8))
    if unassigned:
        ax.scatter(emb[unassigned, 0], emb[unassigned, 1], s=10, alpha=0.25, color="lightgray", label="unassigned")
    for cid, cluster in enumerate(clusters, start=1):
        arr = np.array(cluster, dtype=int)
        medoid = medoid_index(distance, cluster)
        ax.scatter(emb[arr, 0], emb[arr, 1], s=32, alpha=0.85, color=colors[cid - 1], label=f"C{cid} n={len(cluster)}")
        ax.scatter(
            emb[medoid, 0],
            emb[medoid, 1],
            s=110,
            marker="*",
            color=colors[cid - 1],
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.set_title(f"Variable tight/far clusters: {len(clusters)} clusters")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6, ncol=3, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "variable_tight_groups_mds.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster patients into variable-size tight/far groups from a PINN distance matrix."
    )
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument(
        "--within-list",
        type=parse_float_list,
        default=parse_float_list("0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5"),
        help="Complete-link diameter thresholds to scan.",
    )
    parser.add_argument(
        "--center-list",
        type=parse_float_list,
        default=parse_float_list("0"),
        help="Minimum medoid-to-medoid separation thresholds to scan.",
    )
    parser.add_argument(
        "--center-action",
        choices=("none", "reject", "merge"),
        default="none",
        help="How to handle clusters whose medoids are closer than center-list values.",
    )
    parser.add_argument(
        "--min-assigned-frac",
        type=float,
        default=0.85,
        help="Prefer selected configs assigning at least this fraction when possible.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.min_cluster_size < 2:
        raise SystemExit("--min-cluster-size must be at least 2")
    if not (0.0 <= args.min_assigned_frac <= 1.0):
        raise SystemExit("--min-assigned-frac must be between 0 and 1")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    distance = np.load(args.in_dir / "distance_matrix.npy").astype(np.float64)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    pids = np.load(args.in_dir / "valid_pids.npy")
    n = len(pids)
    upper = distance[np.triu_indices(n, k=1)]
    median_pair = float(np.median(upper)) if len(upper) else 1.0

    search_rows: list[dict[str, object]] = []
    selected_by_key: dict[tuple[float, float], tuple[list[list[int]], list[list[int]], list[list[int]]]] = {}
    partitions_by_within: dict[float, list[list[int]]] = {}

    for within_t in args.within_list:
        partitions_by_within[within_t] = complete_linkage_clusters(distance, within_t)
        for center_d in args.center_list:
            if args.center_action == "none":
                clusters = [c for c in partitions_by_within[within_t] if len(c) >= args.min_cluster_size]
                too_small = [c for c in partitions_by_within[within_t] if len(c) < args.min_cluster_size]
                rejected_close = []
            elif args.center_action == "merge":
                clusters, too_small, rejected_close = merge_close_clusters(
                    distance,
                    partitions_by_within[within_t],
                    min_cluster_size=args.min_cluster_size,
                    min_center_dist=center_d,
                )
            else:
                clusters, too_small, rejected_close = select_far_clusters(
                    distance,
                    partitions_by_within[within_t],
                    min_cluster_size=args.min_cluster_size,
                    min_center_dist=center_d,
                )
            row = summarize_config(
                distance,
                clusters,
                n_patients=n,
                within_t=within_t,
                center_d=center_d,
                median_pairwise_distance=median_pair,
            )
            row["small_cluster_patients"] = int(sum(len(c) for c in too_small))
            row["close_rejected_patients"] = int(sum(len(c) for c in rejected_close))
            search_rows.append(row)
            selected_by_key[(within_t, center_d)] = (clusters, too_small, rejected_close)

    best = choose_best(search_rows, args.min_assigned_frac)
    best_key = (float(best["within_t"]), float(best["center_d"]))
    clusters, too_small, rejected_close = selected_by_key[best_key]
    assigned = sorted(idx for c in clusters for idx in c)
    assigned_set = set(assigned)
    unassigned = sorted(set(range(n)) - assigned_set)

    cluster_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cid, cluster in enumerate(clusters):
        m = medoid_index(distance, cluster)
        mean_within = cluster_mean_pairwise(distance, cluster)
        diameter = cluster_diameter(distance, cluster)
        for idx in sorted(cluster, key=lambda i: int(pids[i])):
            cluster_rows.append(
                {
                    "cluster": cid,
                    "pid": int(pids[idx]),
                    "role": "center" if idx == m else "member",
                    "within_mean": mean_within,
                    "max_within_distance": diameter,
                    "distance_to_medoid": float(distance[m, idx]),
                }
            )
        summary_rows.append(
            {
                "cluster": cid,
                "cluster_id": cid + 1,
                "n_patients": len(cluster),
                "medoid_patient": int(pids[m]),
                "within_mean": mean_within,
                "max_within_distance": diameter,
                "patient_ids": ";".join(str(int(pids[i])) for i in sorted(cluster, key=lambda x: int(pids[x]))),
            }
        )

    unassigned_rows = []
    for idx in unassigned:
        reason = "small_cluster_or_noise"
        for c in rejected_close:
            if idx in c:
                reason = "medoid_too_close_to_accepted_cluster"
                break
        unassigned_rows.append({"pid": int(pids[idx]), "reason": reason})

    search_fields = [
        "within_t",
        "center_d",
        "n_clusters",
        "assigned_patients",
        "assigned_frac",
        "unassigned_patients",
        "cluster_sizes",
        "min_cluster_size",
        "median_cluster_size",
        "max_cluster_size",
        "mean_intra_distance",
        "max_intra_distance",
        "min_medoid_distance",
        "mean_medoid_distance",
        "min_inter_patient_distance",
        "mean_nearest_inter_patient_distance",
        "mean_medoid_over_mean_intra",
        "min_medoid_over_max_intra",
        "silhouette",
        "score",
        "small_cluster_patients",
        "close_rejected_patients",
    ]
    write_rows(out_dir / "search_summary.csv", search_rows, search_fields)
    write_rows(
        out_dir / "variable_group_assignments.csv",
        cluster_rows,
        ["cluster", "pid", "role", "within_mean", "max_within_distance", "distance_to_medoid"],
    )
    write_rows(
        out_dir / "cluster_assignments.csv",
        cluster_rows,
        ["cluster", "pid", "role", "within_mean", "max_within_distance", "distance_to_medoid"],
    )
    write_rows(
        out_dir / "cluster_quality_summary.csv",
        summary_rows,
        ["cluster", "cluster_id", "n_patients", "medoid_patient", "within_mean", "max_within_distance", "patient_ids"],
    )
    write_rows(out_dir / "unassigned_patients.csv", unassigned_rows, ["pid", "reason"])
    with (out_dir / "selected_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected": best,
                "n_input_patients": n,
                "min_cluster_size": args.min_cluster_size,
                "within_list": args.within_list,
                "center_list": args.center_list,
                "center_action": args.center_action,
                "min_assigned_frac": args.min_assigned_frac,
            },
            f,
            indent=2,
        )

    emb_path = args.in_dir / "mds_embedding.npy"
    if not args.skip_plots and emb_path.is_file() and clusters:
        plot_clusters(out_dir, distance, np.load(emb_path), clusters, unassigned)

    print(
        "[variable-groups] selected "
        f"within_t={best['within_t']} center_d={best['center_d']} "
        f"clusters={best['n_clusters']} assigned={best['assigned_patients']}/{n} "
        f"mean_intra={float(best['mean_intra_distance']):.3f} "
        f"mean_medoid={float(best['mean_medoid_distance']):.3f} "
        f"ratio={float(best['mean_medoid_over_mean_intra']):.3f}",
        flush=True,
    )
    print(f"[variable-groups] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
