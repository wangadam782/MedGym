"""Normalize PINN clustering outputs into workflow-ready files.

Accepted input formats include:

  - ``cluster,pid,role`` from ``scripts/data_processing/find_tight_groups.py``
  - ``cluster_id,patient_id`` used by online evaluation scripts

The output directory contains:

  - ``cluster_map.csv``: columns ``patient_id,cluster_id,role,distance_to_medoid``
  - ``patient_ids_cluster_order.txt``: one representative/test patient per cluster
  - ``patient_cluster_map.env``: shell assignment for offline scripts
  - ``cluster_<id>_patients.txt``: all patients in each cluster
  - ``cluster_<id>_train_patients.txt``: all non-representative patients
  - ``patient_split_medrl_algorithms_final_mixed.csv``: train/test split used by cluster-pooled RL
  - ``test_patient_ids.txt``: same representatives as patient_ids_cluster_order.txt
  - ``cluster_summary.csv``: counts and representative patient IDs

When ``distance_to_medoid`` is available, the test patient for each cluster is
the patient with the smallest distance.  For outputs from
``find_tight_groups.py`` this is the cluster center/medoid, whose distance is 0.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PATIENT_COLUMNS = ("patient_id", "pid", "icu_id", "stay_id")
CLUSTER_COLUMNS = ("cluster_id", "cluster", "group")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    raise KeyError(f"None of these columns were found: {', '.join(names)}")


def normalize_rows(rows: list[dict[str, str]], renumber: bool) -> list[dict[str, object]]:
    raw: list[tuple[int, int, str, str]] = []
    for row in rows:
        patient_id = int(float(first_existing(row, PATIENT_COLUMNS)))
        cluster_raw = int(float(first_existing(row, CLUSTER_COLUMNS)))
        role = str(row.get("role") or "")
        distance_to_medoid = str(
            row.get("distance_to_medoid") or row.get("distance_to_center") or ""
        )
        raw.append((patient_id, cluster_raw, role, distance_to_medoid))

    if renumber:
        cluster_map = {old: idx + 1 for idx, old in enumerate(sorted({c for _, c, _, _ in raw}))}
    else:
        cluster_map = {old: old for _, old, _, _ in raw}

    normalized = [
        {
            "patient_id": patient_id,
            "cluster_id": cluster_map[cluster_raw],
            "role": role,
            "distance_to_medoid": distance_to_medoid,
        }
        for patient_id, cluster_raw, role, distance_to_medoid in raw
    ]
    return sorted(normalized, key=lambda r: (int(r["cluster_id"]), int(r["patient_id"])))


def _distance_value(row: dict[str, object]) -> float | None:
    raw = row.get("distance_to_medoid")
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def representative_for_cluster(rows: list[dict[str, object]]) -> int:
    with_dist = [(r, _distance_value(r)) for r in rows]
    finite_dist = [(r, d) for r, d in with_dist if d is not None]
    if finite_dist:
        selected = sorted(finite_dist, key=lambda item: (float(item[1]), int(item[0]["patient_id"])))[0][0]
        return int(selected["patient_id"])

    centers = [r for r in rows if str(r.get("role", "")).lower() == "center"]
    selected = centers[0] if centers else sorted(rows, key=lambda r: int(r["patient_id"]))[0]
    return int(selected["patient_id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert clustering assignment CSVs into cluster maps and patient lists."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--keep-cluster-labels",
        action="store_true",
        help="Do not remap sorted cluster labels to 1..N.",
    )
    args = parser.parse_args()

    rows = normalize_rows(read_rows(args.input), renumber=not args.keep_cluster_labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_rows(
        args.out_dir / "cluster_map.csv",
        rows,
        ["patient_id", "cluster_id", "role", "distance_to_medoid"],
    )

    by_cluster: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_cluster[int(row["cluster_id"])].append(row)

    summary_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    recommended_rows: list[dict[str, object]] = []
    patient_cluster_entries: list[str] = []
    rep_ids: list[int] = []
    for cluster_id in sorted(by_cluster):
        cluster_rows = by_cluster[cluster_id]
        patient_ids = sorted(int(row["patient_id"]) for row in cluster_rows)
        row_by_pid = {int(row["patient_id"]): row for row in cluster_rows}
        representative = representative_for_cluster(cluster_rows)
        rep_ids.append(representative)
        patient_cluster_entries.append(f"{representative}:{cluster_id}")
        (args.out_dir / f"cluster_{cluster_id}_patients.txt").write_text(
            "\n".join(str(pid) for pid in patient_ids) + "\n",
            encoding="utf-8",
        )
        train_ids = [pid for pid in patient_ids if pid != representative]
        (args.out_dir / f"cluster_{cluster_id}_train_patients.txt").write_text(
            "\n".join(str(pid) for pid in train_ids) + "\n",
            encoding="utf-8",
        )
        for pid in patient_ids:
            is_rep = pid == representative
            split_rows.append(
                {
                    "cluster_id": cluster_id,
                    "patient_id": pid,
                    "split": "test" if is_rep else "train",
                    "active_for_run": "False" if is_rep else "True",
                    "medoid_patient": representative,
                }
            )
            recommended_rows.append(
                {
                    "patient_id": pid,
                    "cluster_id": cluster_id,
                    "cluster_size": len(patient_ids),
                    "is_medoid": "True" if is_rep else "False",
                    "distance_to_medoid": row_by_pid[pid].get("distance_to_medoid") or (0.0 if is_rep else ""),
                }
            )
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "n_patients": len(patient_ids),
                "n_train": len(train_ids),
                "n_test": 1,
                "test_patient_id": representative,
                "medoid_patient": representative,
                "patient_ids": ";".join(str(pid) for pid in patient_ids),
            }
        )

    write_rows(
        args.out_dir / "recommended_clustering.csv",
        recommended_rows,
        ["patient_id", "cluster_id", "cluster_size", "is_medoid", "distance_to_medoid"],
    )
    write_rows(
        args.out_dir / "patient_split_medrl_algorithms_final_mixed.csv",
        split_rows,
        ["cluster_id", "patient_id", "split", "active_for_run", "medoid_patient"],
    )
    write_rows(
        args.out_dir / "cluster_summary.csv",
        summary_rows,
        [
            "cluster_id",
            "n_patients",
            "n_train",
            "n_test",
            "test_patient_id",
            "medoid_patient",
            "patient_ids",
        ],
    )
    (args.out_dir / "patient_ids_cluster_order.txt").write_text(
        "\n".join(str(pid) for pid in rep_ids) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "test_patient_ids.txt").write_text(
        "\n".join(str(pid) for pid in rep_ids) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "patient_cluster_map.env").write_text(
        'PATIENT_CLUSTER_MAP="' + " ".join(patient_cluster_entries) + '"\n',
        encoding="utf-8",
    )
    print(f"[cluster-map] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
