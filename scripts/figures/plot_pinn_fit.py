#!/usr/bin/env python3
"""Reproduce Fig 3, Fig 4, and Fig 7 for patient 202347.

Prerequisites
-------------
Download checkpoints-case-pinn from HuggingFace and extract at the repo root:
  checkpoints-case-pinn/pid202347.csv
  checkpoints-case-pinn/pinn/train70/patient_202347/
  checkpoints-case-pinn/pinn/individual/patient_202347/
  checkpoints-case-pinn/pinn/population/
  checkpoints-case-pinn/pinn/cluster_pooled/cluster_1/
  checkpoints-case-pinn/online/{individual,population,cluster_pooled}/...

Usage
-----
  cd medrl-tacos
  python scripts/figures/plot_pinn_fit.py            # all three figures
  python scripts/figures/plot_pinn_fit.py --fig 3    # Fig 3: PINN fit
  python scripts/figures/plot_pinn_fit.py --fig 4    # Fig 4: RL comparison
  python scripts/figures/plot_pinn_fit.py --fig 7    # Fig 7: Ind PINN (train70)

Outputs
-------
  out/fig3_pid202347/pinn_fit_pid202347.png
  out/fig4_pid202347/rl_lagrangian_trpo_pid202347_K20.png
  out/fig7_pid202347/fig7_pid202347_ind_only.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

from scripts.figures.plot_global_vs_perpatient import (
    plot_pinn_fit_combined,
    plot_rl_comparison_combined,
)

# ── shared constants ──────────────────────────────────────────────────────────
ICU_ID      = 202347
K           = 20
FILE_PREFIX = "lagrangian_trpo"
CASE_DATA   = PROJECT_ROOT / "checkpoints-case-pinn"

_ONLINE = CASE_DATA / "online"
_IND_RL = _ONLINE / "individual" / f"patient_{ICU_ID}" / FILE_PREFIX / f"K{K}"
_POP_RL = _ONLINE / "population" / FILE_PREFIX / f"K{K}"
_CLU_RL = _ONLINE / "cluster_pooled" / "cluster_1" / FILE_PREFIX / f"K{K}"


# ── Fig 3: PINN fit (Pop + Clu + Ind) ────────────────────────────────────────
def plot_fig3() -> None:
    csv      = CASE_DATA / "pid202347.csv"
    ind_dir  = CASE_DATA / "pinn" / "individual" / "patient_202347"
    pop_dir  = CASE_DATA / "pinn" / "population"
    clu_dir  = CASE_DATA / "pinn" / "cluster_pooled" / "cluster_1"
    save_dir = PROJECT_ROOT / "out" / "fig3_pid202347"
    save     = save_dir / "pinn_fit_pid202347.png"

    for label, path in [
        ("CSV",        csv),
        ("Ind PINN",   ind_dir / "pinn.pt"),
        ("Ind scales", ind_dir / "scales.npy"),
        ("Pop PINN",   pop_dir / "pinn.pt"),
        ("Pop scales", pop_dir / "scales.npy"),
        ("Clu PINN",   clu_dir / "pinn.pt"),
        ("Clu scales", clu_dir / "scales.npy"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Fig 3] patient={ICU_ID}  Pop + Clu (cluster_1) + Ind")

    plot_pinn_fit_combined(
        csv_path               = str(csv),
        icu_id                 = ICU_ID,
        global_pinn_path       = pop_dir / "pinn.pt",
        global_scales_path     = pop_dir / "scales.npy",
        perpatient_pinn_path   = ind_dir / "pinn.pt",
        perpatient_scales_path = None,
        cluster_pinn_path      = clu_dir / "pinn.pt",
        cluster_scales_path    = clu_dir / "scales.npy",
        save_path              = str(save),
    )
    print(f"[Done]  {save}")


# ── Fig 4: RL comparison (Pop + Clu + Ind, K=20) ─────────────────────────────
def plot_fig4() -> None:
    npz_paths = {
        "ind_vardt": _IND_RL / "vardt" / "eval" / "aggregated.npz",
        "ind_fixdt": _IND_RL / "fixdt" / "eval" / "aggregated.npz",
        "pop_vardt": _POP_RL / "vardt" / "eval" / "aggregated.npz",
        "pop_fixdt": _POP_RL / "fixdt" / "eval" / "aggregated.npz",
        "clu_vardt": _CLU_RL / "vardt" / "eval" / "aggregated.npz",
        "clu_fixdt": _CLU_RL / "fixdt" / "eval" / "aggregated.npz",
    }
    missing = [p for p in npz_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing aggregated.npz files:\n" + "\n".join(f"  {p}" for p in missing)
        )

    save_dir = PROJECT_ROOT / "out" / "fig4_pid202347"
    save     = save_dir / f"rl_{FILE_PREFIX}_pid{ICU_ID}_K{K}.png"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Fig 4] patient={ICU_ID}  algo={FILE_PREFIX}  K={K}")

    plot_rl_comparison_combined(
        icu_id             = ICU_ID,
        global_root        = "",
        perpatient_root    = "",
        K                  = K,
        save_path          = str(save),
        file_prefix        = FILE_PREFIX,
        include_cluster    = True,
        include_population = True,
        npz_paths          = npz_paths,
    )
    print(f"[Done]  {save}")
    print(f"        subplots → {save.parent / save.stem}/")


# ── Fig 7: Ind PINN (train70) ─────────────────────────────────────────────────
def plot_fig7() -> None:
    csv      = CASE_DATA / "pid202347.csv"
    ind_dir  = CASE_DATA / "pinn" / "train70" / "patient_202347"
    save_dir = PROJECT_ROOT / "out" / "fig7_pid202347"
    save     = save_dir / "fig7_pid202347_ind_only.png"

    for label, path in [
        ("CSV",        csv),
        ("Ind PINN",   ind_dir / "pinn.pt"),
        ("Ind scales", ind_dir / "scales.npy"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Fig 7] patient={ICU_ID}  Ind-only (train70)  train_ratio=0.7")

    plot_pinn_fit_combined(
        csv_path               = str(csv),
        icu_id                 = ICU_ID,
        global_pinn_path       = None,
        global_scales_path     = None,
        perpatient_pinn_path   = ind_dir / "pinn.pt",
        perpatient_scales_path = ind_dir / "scales.npy",
        cluster_pinn_path      = None,
        cluster_scales_path    = None,
        save_path              = str(save),
        figsize                = (20.0, 6.5),
        title                  = "",
        annotation             = None,
        axis_label_size        = 22.0,
        tick_label_size        = 18.0,
        legend_size            = 16.0,
        color_true             = "#666666",
        color_ind              = "red",
        label_true             = "True",
        label_ind              = "Ind PINN",
        legend_panel           = True,
        sofa_ylabel            = "SOFA",
        train_ratio            = 0.7,
    )
    print(f"[Done]  {save}")


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Fig 3 / Fig 4 / Fig 7 for patient 202347"
    )
    parser.add_argument(
        "--fig", choices=["3", "4", "7", "all"], default="all",
        help="Which figure to plot (default: all)",
    )
    args = parser.parse_args()

    if args.fig in ("3", "all"):
        plot_fig3()
    if args.fig in ("4", "all"):
        plot_fig4()
    if args.fig in ("7", "all"):
        plot_fig7()


if __name__ == "__main__":
    main()
