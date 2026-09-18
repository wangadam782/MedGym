#!/usr/bin/env python3
"""
compute_pinn_pairwise_dist.py
=============================

CLI script to compute bidirectional PINN-to-PINN distances for all patients with a
PINN checkpoint, or for the subset given by ``--patient-ids-csv``.

Main outputs under ``<out-dir>``
--------------------------------
<out-dir>/distance_matrix.npy
<out-dir>/valid_pids.npy
<out-dir>/distance_matrix.csv
<out-dir>/nearest_neighbors.csv
<out-dir>/global_std.npy
<out-dir>/progress.log

Defaults
--------
--csv         data/mimic_pinn_v4_filtered.csv
--patient-dir results/pinn/individual
--out-dir     results_pinn_pairwise_dist
--k-steps     20

Patient directories may use either layout:
  results/pinn/individual/<pid>/pinn.pt
  results/pinn/individual/patient_<pid>/pinn.pt

"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

# Heavy scientific imports are loaded inside main() so that --help returns
# immediately and wrapper scripts can inspect the CLI quickly.

warnings.filterwarnings("ignore")

ID_COLUMNS = ("pid", "icu_id", "patient_id", "stay_id")
ZERO_IS_NAN_FALLBACK = {"SpO2", "PaO2", "Bilirubin", "GCS"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute bidirectional PINN-to-PINN pairwise distances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv", "--csv-path", dest="csv", type=Path,
        default=Path("data/mimic_pinn_v4_filtered.csv"),
        help="Patient time-series CSV used for pairwise distance calculation.",
    )
    p.add_argument(
        "--patient-dir", "--pinn-root", "--in-dir", dest="patient_dir", type=Path,
        default=Path("results/pinn/individual"),
        help="Directory containing individual PINN outputs.",
    )
    p.add_argument(
        "--out-dir", "--output-dir", dest="out_dir", type=Path,
        default=Path("results_pinn_pairwise_dist"),
        help="Output directory for distance matrix and artifacts.",
    )
    p.add_argument(
        "--patient-ids-csv", "--ids-csv", "--pid-csv", dest="patient_ids_csv",
        type=Path, default=None,
        help="Optional pid-only CSV used to restrict target patients.",
    )
    p.add_argument(
        "--repo-root", type=Path, default=Path.cwd(),
        help="Repository root. Used to resolve Python import paths.",
    )
    p.add_argument(
        "--module-root", action="append", type=Path, default=[],
        help="Additional directory to add to sys.path before importing data.config. Can be repeated.",
    )
    p.add_argument("--k-steps", type=int, default=20, help="Rollout steps used for distance calculation.")
    p.add_argument("--max-patients", type=int, default=None, help="Optional cap for smoke tests.")
    p.add_argument(
        "--global-std-path", type=Path, default=None,
        help="Optional path to load/save gstd. If omitted, <out-dir>/global_std.npy is used.",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for model rollout.",
    )
    p.add_argument(
        "--progress-every", type=int, default=100,
        help="Progress logging interval.",
    )
    p.add_argument("--skip-mds", action="store_true", help="Skip MDS embedding plot.")
    p.add_argument("--skip-plots", action="store_true", help="Skip all plots.")
    return p.parse_args()


def add_import_paths(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve()
    candidates = [
        root,
        here,
        here.parent,
        *[p.resolve() for p in args.module_root],
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def import_data_config() -> tuple:
    try:
        from data.config import (  # type: ignore
            ZERO_IS_NAN,
            action_dim,
            action_features,
            state_dim,
            state_features,
        )
        return ZERO_IS_NAN, int(action_dim), list(action_features), int(state_dim), list(state_features)
    except Exception as exc:
        raise SystemExit(
            "Could not import data.config. Run from repo root or pass --repo-root / --module-root.\n"
            f"Import error: {exc}"
        )


def read_pid_csv(path: Path) -> list[int]:
    df = pd.read_csv(path, comment="#")
    for col in ID_COLUMNS:
        if col in df.columns:
            return sorted({int(x) for x in df[col].dropna().tolist()})
    col = df.columns[0]
    return sorted({int(x) for x in df[col].dropna().tolist()})


def pid_from_dir_name(name: str) -> int | None:
    if name.isdigit():
        return int(name)
    m = re.fullmatch(r"patient_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def discover_pinn_paths(patient_dir: Path, allowed_ids: set[int] | None = None) -> list[tuple[int, Path]]:
    if not patient_dir.is_dir():
        raise SystemExit(f"Patient directory not found: {patient_dir}")

    found: dict[int, Path] = {}
    for d in sorted(patient_dir.iterdir()):
        if not d.is_dir():
            continue
        pid = pid_from_dir_name(d.name)
        if pid is None:
            continue
        if allowed_ids is not None and pid not in allowed_ids:
            continue
        for fname in ("pinn.pt", "model.pt"):
            pt = d / fname
            if pt.is_file():
                found[pid] = pt
                break

    return sorted(found.items(), key=lambda x: x[0])


def make_pinn_class(state_dim: int, action_dim: int):
    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.LayerNorm(128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, state_dim),
            )
            s = torch.ones(state_dim)
            if state_dim > 2:
                s[2] = 0.01
            self.log_scale = nn.Parameter(torch.log(s))

        def forward(self, x, a):
            raw = self.net(torch.cat([x, a], -1))
            scale = torch.nn.functional.softplus(self.log_scale).unsqueeze(0)
            return raw * scale

    return PINN


def normalize_state_dict_keys(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model.", "pinn."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def load_pinn_model(PINN, path: Path, device: torch.device):
    model = PINN().to(device)
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "pinn_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    try:
        model.load_state_dict(obj)
    except Exception:
        model.load_state_dict(normalize_state_dict_keys(obj))
    model.eval()
    return model


def prep_patient_real(
    df_p: pd.DataFrame,
    *,
    state_features: list[str],
    action_features: list[str],
    zero_is_nan: Iterable[str],
):
    """Return real-scale sv_imp, av_raw, scales dict, mask, and dt_arr."""
    zero_is_nan = set(zero_is_nan)
    df_p = df_p.sort_values("hours")

    sv_raw = df_p[state_features].values.astype(np.float32)
    for i, f in enumerate(state_features):
        if f in zero_is_nan:
            sv_raw[:, i] = np.where(sv_raw[:, i] == 0, np.nan, sv_raw[:, i])

    mask = (~np.isnan(sv_raw)).astype(np.float32)
    mean_ = np.nanmean(sv_raw, 0)
    mean_ = np.where(np.isnan(mean_), 0.0, mean_).astype(np.float32)
    std_ = np.nanstd(sv_raw, 0)
    std_ = np.where(np.isnan(std_) | (std_ == 0), 1.0, std_).astype(np.float32)
    sv_imp = np.where(np.isnan(sv_raw), mean_, sv_raw).astype(np.float32)

    av_raw = df_p[action_features].values.astype(np.float32)
    av_raw = np.clip(np.where(np.isnan(av_raw), 0.0, av_raw), 0.0, None).astype(np.float32)
    ascl = np.nanmax(np.log1p(av_raw), 0).astype(np.float32)
    ascl[ascl == 0] = 1.0

    hrs = df_p["hours"].values.astype(np.float32)
    dt_arr = np.clip(np.diff(hrs), 1e-3, 24.0).astype(np.float32)
    return sv_imp, av_raw, {"mean": mean_, "std": std_, "ascl": ascl}, mask, dt_arr


def batch_rollout_real(
    pinn_j,
    sc_j: dict,
    all_sv_imp: np.ndarray,
    all_av_raw: np.ndarray,
    all_dt: np.ndarray,
    *,
    k_steps: int,
    device: torch.device,
):
    """Roll out PINN_j from every patient's initial state for k_steps; return real-scale trajectories."""
    mean_j = torch.tensor(sc_j["mean"], dtype=torch.float32, device=device)
    std_j = torch.tensor(sc_j["std"], dtype=torch.float32, device=device)
    ascl_j = torch.tensor(sc_j["ascl"], dtype=torch.float32, device=device)

    x_norm = (torch.tensor(all_sv_imp, dtype=torch.float32, device=device) - mean_j) / std_j
    a_norm = torch.log1p(torch.tensor(all_av_raw, dtype=torch.float32, device=device)) / ascl_j
    dt_t = torch.tensor(all_dt, dtype=torch.float32, device=device)

    traj = [x_norm.unsqueeze(1)]
    x_t = x_norm.clone()
    with torch.no_grad():
        for k in range(k_steps):
            dx = pinn_j(x_t, a_norm[:, k])
            x_t = x_t + dx * dt_t[:, k:k + 1]
            traj.append(x_t.unsqueeze(1))

    traj_norm = torch.cat(traj, dim=1)
    traj_real = (traj_norm * std_j + mean_j).detach().cpu().numpy()
    return traj_real.astype(np.float32)


def prepare_patients(
    *,
    df_all: pd.DataFrame,
    pinn_items: list[tuple[int, Path]],
    k_steps: int,
    state_dim: int,
    action_dim: int,
    state_features: list[str],
    action_features: list[str],
    zero_is_nan: Iterable[str],
) -> tuple[list[int], list[Path], np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray, list[int]]:
    sv_list: list[np.ndarray] = []
    av_list: list[np.ndarray] = []
    dt_list: list[np.ndarray] = []
    scales: list[dict] = []
    masks: list[np.ndarray] = []
    valid_pids: list[int] = []
    valid_paths: list[Path] = []
    skipped_no_csv: list[int] = []

    for pid, pinn_path in pinn_items:
        df_p = df_all[df_all["icu_id"].astype(int) == int(pid)]
        if len(df_p) < 2:
            skipped_no_csv.append(pid)
            continue

        sv_imp, av_raw, sc, mask, dt_arr = prep_patient_real(
            df_p,
            state_features=state_features,
            action_features=action_features,
            zero_is_nan=zero_is_nan,
        )
        obs_count = mask.sum(1)
        start_idx = int(np.argmax(obs_count >= state_dim / 2)) if (obs_count >= state_dim / 2).any() else 0

        sv0 = sv_imp[start_idx].astype(np.float32)

        av_slice = av_raw[start_idx:start_idx + k_steps]
        if len(av_slice) == 0:
            av_slice = np.zeros((1, action_dim), dtype=np.float32)
        pad_len = k_steps - len(av_slice)
        if pad_len > 0:
            av_slice = np.concatenate([av_slice, np.tile(av_slice[-1:], (pad_len, 1))], axis=0)

        dt_slice = dt_arr[start_idx:start_idx + k_steps]
        pad_len = k_steps - len(dt_slice)
        if pad_len > 0:
            last_dt = dt_slice[-1:] if len(dt_slice) > 0 else np.ones(1, dtype=np.float32)
            dt_slice = np.concatenate([dt_slice, np.tile(last_dt, pad_len)], axis=0)

        sv_list.append(sv0)
        av_list.append(av_slice.astype(np.float32))
        dt_list.append(dt_slice.astype(np.float32))
        scales.append(sc)
        masks.append(mask[start_idx].astype(np.float32))
        valid_pids.append(int(pid))
        valid_paths.append(pinn_path)

    if not valid_pids:
        raise SystemExit("No patients remained after matching PINN dirs with CSV rows.")

    return (
        valid_pids,
        valid_paths,
        np.stack(sv_list).astype(np.float32),
        np.stack(av_list).astype(np.float32),
        np.stack(dt_list).astype(np.float32),
        scales,
        np.stack(masks).astype(np.float32),
        skipped_no_csv,
    )


def save_plots_and_tables(
    *,
    out_dir: Path,
    D: np.ndarray,
    valid_pids: list[int],
    skip_plots: bool,
    skip_mds: bool,
) -> None:
    N = len(valid_pids)
    mean_dist = D.mean(1)

    # nearest_neighbors.csv
    k_nn = min(10, max(0, N - 1))
    nn_data: dict[str, list] = {
        "pid": valid_pids,
        "mean_dist": mean_dist.tolist(),
    }
    if k_nn > 0:
        nn_idx = np.argsort(D, axis=1)[:, 1:k_nn + 1]
        nn_dist = np.sort(D, axis=1)[:, 1:k_nn + 1]
        for k in range(k_nn):
            nn_data[f"nn{k+1}_pid"] = [valid_pids[nn_idx[i, k]] for i in range(N)]
            nn_data[f"nn{k+1}_dist"] = nn_dist[:, k].tolist()
    pd.DataFrame(nn_data).to_csv(out_dir / "nearest_neighbors.csv", index=False)
    print("  [Saved] nearest_neighbors.csv", flush=True)

    if skip_plots or N < 2:
        return

    upper = D[np.triu_indices(N, k=1)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(upper, bins=100, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(upper), color="red", lw=1.5, label=f"median={np.median(upper):.2f}")
    ax.axvline(2.5, color="orange", lw=1.5, ls="--", label="d=2.5")
    ax.set_xlabel("Bidirectional PINN-to-PINN distance")
    ax.set_ylabel("Count")
    ax.set_title(f"All pairwise distances  N={N}  pairs={len(upper):,}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dist_histogram.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] dist_histogram.png", flush=True)

    top_n = min(200, N)
    sorted_idx = np.argsort(mean_dist)[:top_n]
    D_sub = D[np.ix_(sorted_idx, sorted_idx)]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(D_sub, aspect="auto", cmap="viridis_r", vmin=0, vmax=np.percentile(D_sub, 95))
    plt.colorbar(im, ax=ax, label="distance")
    ax.set_title(f"Distance heatmap (top-{top_n} most similar patients)")
    ax.set_xlabel("Patient index (sorted)")
    ax.set_ylabel("Patient index (sorted)")
    plt.tight_layout()
    fig.savefig(out_dir / "dist_heatmap_top200.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] dist_heatmap_top200.png", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_d = np.sort(upper)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, lw=1.5, color="steelblue")
    ax.axvline(2.5, color="orange", lw=1.5, ls="--", label="d=2.5")
    frac_below = float((upper < 2.5).mean()) if len(upper) else 0.0
    ax.set_title(f"CDF of pairwise distances ({frac_below*100:.1f}% pairs have d<2.5)")
    ax.set_xlabel("distance")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dist_cdf.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] dist_cdf.png", flush=True)

    if skip_mds or N < 3:
        return

    print("  Running MDS ...", flush=True)
    t_mds = time.time()
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_jobs=-1)
    emb = mds.fit_transform(D)
    np.save(out_dir / "mds_embedding.npy", emb)
    print(f"  MDS done in {time.time() - t_mds:.1f}s", flush=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(emb[:, 0], emb[:, 1], s=10, alpha=0.5, c=mean_dist, cmap="plasma", linewidths=0)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(mean_dist.min(), mean_dist.max()))
    plt.colorbar(sm, ax=ax, label="Mean distance to all others")
    ax.set_title(f"MDS 2-D projection of all {N} patients")
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / "mds_scatter_all.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] mds_scatter_all.png", flush=True)


def main() -> None:
    args = parse_args()

    # Import heavy libraries only after argparse has handled --help.
    global np, pd, torch, nn, plt, MDS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.manifold import MDS

    add_import_paths(args)
    zero_is_nan, action_dim, action_features, state_dim, state_features = import_data_config()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    if progress_path.exists():
        progress_path.unlink()

    if not args.csv.is_file():
        raise SystemExit(
            f"CSV not found: {args.csv}\n"
            "Pairwise computation requires a trajectory CSV."
        )

    allowed_ids = set(read_pid_csv(args.patient_ids_csv)) if args.patient_ids_csv else None
    if allowed_ids is not None:
        print(f"Restricting to pid CSV: {args.patient_ids_csv}  ({len(allowed_ids)} patients)", flush=True)

    pinn_items = discover_pinn_paths(args.patient_dir, allowed_ids)
    if args.max_patients is not None:
        pinn_items = pinn_items[:args.max_patients]
    if not pinn_items:
        raise SystemExit(f"No pinn.pt/model.pt found under {args.patient_dir}")

    print(f"PINN checkpoints found: {len(pinn_items)}", flush=True)
    t_prep = time.time()
    print(f"Loading CSV: {args.csv}", flush=True)
    df_all = pd.read_csv(args.csv)
    if "icu_id" not in df_all.columns or "hours" not in df_all.columns:
        raise SystemExit(f"Expected columns 'icu_id' and 'hours' in {args.csv}")
    df_all = df_all.copy()
    df_all["icu_id"] = df_all["icu_id"].astype(np.int64)
    (
        valid_pids,
        pinn_paths,
        sv_all,
        av_all,
        dt_all,
        scales,
        masks,
        skipped_no_csv,
    ) = prepare_patients(
        df_all=df_all,
        pinn_items=pinn_items,
        k_steps=args.k_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        state_features=state_features,
        action_features=action_features,
        zero_is_nan=zero_is_nan or ZERO_IS_NAN_FALLBACK,
    )
    N = len(valid_pids)
    print(f"Data prep done: {N} patients  ({time.time() - t_prep:.1f}s)", flush=True)
    if skipped_no_csv:
        print(f"  Skipped (no CSV rows): {len(skipped_no_csv)} patients (first 20: {skipped_no_csv[:20]})", flush=True)

    np.save(out_dir / "valid_pids.npy", np.array(valid_pids, dtype=np.int64))
    pd.DataFrame({"pid": valid_pids, "pinn_path": [str(p) for p in pinn_paths]}).to_csv(
        out_dir / "valid_pids.csv", index=False
    )

    gstd_path = args.global_std_path or (out_dir / "global_std.npy")
    if gstd_path.exists():
        gstd = np.load(gstd_path).astype(np.float32)
        print(f"gstd: loaded from {gstd_path}", flush=True)
    else:
        gstd = sv_all.std(0).astype(np.float32) + 1e-8
        gstd_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(gstd_path, gstd)
        print(f"gstd: computed and saved to {gstd_path}", flush=True)
    print(f"gstd values: {gstd.round(3)}", flush=True)

    # Expect state_features like [SpO2, PaO2, Bilirubin, GCS, Urine_Step, Lactate].
    # Double weight on PaO2, GCS, Lactate (same as the legacy script).
    high_weight_features = {"PaO2", "GCS", "Lactate"}
    clin_w = np.array([2.0 if f in high_weight_features else 1.0 for f in state_features], dtype=np.float32)
    print(f"Clinical feature weights: {dict(zip(state_features, clin_w))}", flush=True)

    PINN = make_pinn_class(state_dim, action_dim)

    # Pass 1
    print("\nPass 1: self rollouts (fp_self) ...", flush=True)
    fp_self = np.zeros((N, args.k_steps + 1, state_dim), dtype=np.float32)
    t1 = time.time()
    for ii, (pid, pinn_path) in enumerate(zip(valid_pids, pinn_paths)):
        model = load_pinn_model(PINN, pinn_path, device)
        traj = batch_rollout_real(
            model,
            scales[ii],
            sv_all[ii:ii + 1],
            av_all[ii:ii + 1],
            dt_all[ii:ii + 1],
            k_steps=args.k_steps,
            device=device,
        )
        fp_self[ii] = traj[0]
        if (ii + 1) % args.progress_every == 0 or ii + 1 == N:
            elapsed = time.time() - t1
            remain = elapsed / (ii + 1) * (N - ii - 1)
            msg = f"Pass1 {ii+1}/{N}  elapsed={elapsed:.0f}s  remain≈{remain:.0f}s"
            print("  " + msg, flush=True)
            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(msg + "\n")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    np.save(out_dir / "fp_self.npy", fp_self)
    print(f"Pass 1 finished in {time.time() - t1:.1f}s", flush=True)

    # Pass 2
    print("\nPass 2: full pairwise distance matrix ...", flush=True)
    D_raw = np.zeros((N, N), dtype=np.float32)
    eff_w = (clin_w / (gstd + 1e-8)).astype(np.float32)
    eff_w_sum = float(clin_w.sum())
    t2 = time.time()
    for jj, (pid_j, pinn_path) in enumerate(zip(valid_pids, pinn_paths)):
        model = load_pinn_model(PINN, pinn_path, device)
        fp_j_all = batch_rollout_real(
            model,
            scales[jj],
            sv_all,
            av_all,
            dt_all,
            k_steps=args.k_steps,
            device=device,
        )
        diff = np.abs(fp_j_all - fp_self)
        diff_w = diff * eff_w[None, None, :]
        D_raw[jj] = diff_w.sum(axis=2).mean(axis=1) / eff_w_sum

        if (jj + 1) % args.progress_every == 0 or jj + 1 == N:
            elapsed = time.time() - t2
            remain = elapsed / (jj + 1) * (N - jj - 1)
            msg = f"Pass2 {jj+1}/{N}  elapsed={elapsed:.0f}s  remain≈{remain:.0f}s"
            print("  " + msg, flush=True)
            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(msg + "\n")
        del model, fp_j_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

    np.save(out_dir / "distance_matrix_raw.npy", D_raw)
    D = (D_raw + D_raw.T) / 2.0
    np.fill_diagonal(D, 0.0)
    np.save(out_dir / "distance_matrix.npy", D)
    pd.DataFrame(D, index=valid_pids, columns=valid_pids).to_csv(out_dir / "distance_matrix.csv")
    print(f"Saved distance matrix: {out_dir / 'distance_matrix.npy'}  shape={D.shape}", flush=True)

    if N >= 2:
        upper = D[np.triu_indices(N, k=1)]
        print("\nDistance statistics:", flush=True)
        stats = {}
        for q in [0, 10, 25, 50, 75, 90, 99, 100]:
            stats[f"p{q}"] = float(np.percentile(upper, q))
            print(f"  p{q:3d}: {stats[f'p{q}']:.3f}", flush=True)
    else:
        stats = {}

    save_plots_and_tables(
        out_dir=out_dir,
        D=D,
        valid_pids=valid_pids,
        skip_plots=args.skip_plots,
        skip_mds=args.skip_mds,
    )

    meta = {
        "script": "compute_pinn_pairwise_dist.py",
        "csv": str(args.csv),
        "patient_dir": str(args.patient_dir),
        "patient_ids_csv": str(args.patient_ids_csv) if args.patient_ids_csv else None,
        "out_dir": str(out_dir),
        "n_patients": N,
        "k_steps": int(args.k_steps),
        "device": str(device),
        "state_features": state_features,
        "action_features": action_features,
        "clinical_weights": dict(zip(state_features, [float(x) for x in clin_w])),
        "global_std_path": str(gstd_path),
        "distance_stats": stats,
        "skipped_no_csv": skipped_no_csv,
    }
    with open(out_dir / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nAll done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
