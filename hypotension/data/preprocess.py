"""CSV -> per-patient arrays + scales.

    python -m hypotension.data.preprocess --csv hypotension/data/hypotension_healthgym.csv

Outputs hypotension/data/processed/
    scales.npy          {mean, std, ascl_mean, ascl_std, state_min, state_max}
    patients/P####.npz  t, X (physical units), U (physical units), M (obs mask)
    cohort_long.csv     cleaned long-format table for manual inspection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hypotension.data.config import (
    ACTION_NAMES, CLIP, DEFAULT_CSV, DT_HOURS, PHYSICAL_MAX, PHYSICAL_MIN,
    PROCESSED_ROOT, REFS, STATE_NAMES, UNOBSERVED,
)

RAW = dict(
    pid="PatientID", t="Timepoints",
    map="MAP", sbp="systolic_bp", dbp="diastolic_bp",
    urine="urine", alt="ALT", ast="AST", pao2="PO2",
    lac="lactic_acid", cr="serum_creatinine",
    fluid="fluid_boluses", vaso="vasopressors", fio2="FiO2", gcs="GCS_total",
    m_urine="urine_m", m_hep="ALT_AST_m", m_fio2="FiO2_m", m_gcs="GCS_total_m",
    m_pao2="PO2_m", m_lac="lactic_acid_m", m_cr="serum_creatinine_m",
)
VASO_TRACE = 1e-5      # raw value 1e-06 is a "trace/zero" placeholder


# =============================================================================
def load_and_clean(csv, n_patients=None, weight_kg=REFS.weight_kg, verbose=True):
    df = pd.read_csv(csv).rename(columns={v: k for k, v in RAW.items()})
    df = df.sort_values(["pid", "t"]).reset_index(drop=True)
    if n_patients:
        df = df[df.pid.isin(np.sort(df.pid.unique())[:n_patients])].reset_index(drop=True)

    # ---- State (physical units) ----
    df["MAP"] = df["map"]
    df["PP"] = df.sbp - df.dbp
    df["UO"] = df.urine / weight_kg       # mL/h -> mL/kg/h
    df["Lac"], df["Cr"] = df.lac, df.cr
    df["Hep"], df["AST"] = df.alt, df.ast
    df["PaO2"], df["GCS"] = df.pao2, df.gcs.astype(float)

    # ---- Actions (physical units) ----
    df["Fluids"] = df.fluid / weight_kg / DT_HOURS
    v = np.array(df.vaso, float, copy=True); v[v < VASO_TRACE] = 0.0
    df["Vaso"] = v / weight_kg
    df["FiO2"] = df.fio2.clip(0.21, 1.0)

    # ---- Observation mask ----
    df["m_MAP"] = df["m_PP"] = 1.0
    for tgt, src in [("UO", "m_urine"), ("Lac", "m_lac"), ("Cr", "m_cr"),
                     ("Hep", "m_hep"), ("PaO2", "m_pao2"), ("GCS", "m_gcs")]:
        df[f"m_{tgt}"] = df[src].astype(float)

    nclip = {}
    for c, (lo, hi) in CLIP.items():
        if c in df:
            n = int(((df[c] < lo) | (df[c] > hi)).sum())
            if n: nclip[c] = n
            df[c] = df[c].clip(lo, hi)
    if verbose:
        print(f"[load ] {df.pid.nunique()} patients x {df.groupby('pid').size().unique()} steps")
        print(f"[clip ] out-of-range clipped: {nclip or 'none'}")
    return df


def compute_latents(df, verbose=True):
    """Integrate V (volume) and Ce (vasopressor effect-compartment) from records.

    Both are derived from recorded orders + observed UO — they are not free
    latent variables. V(0) is solved analytically from MAP(0) via Frank-Starling.
    """
    r = REFS; n_sub = 8; h = DT_HOURS / n_sub
    (lV, hV), (lC, hC) = CLIP["V"], CLIP["Ce"]
    Vs_all, Ce_all = [], []
    for _, g in df.groupby("pid", sort=False):
        Ce = float(g.Vaso.iloc[0])
        eff = r.E_vaso * Ce / (Ce + r.K_vaso)
        sv = np.clip((g.MAP.iloc[0] - r.MAP_base - eff) / r.E_preload + 1.0,
                     0.05, 1.0 + r.K_fs - 1e-2)
        V = float(np.clip(r.K_fs * sv / max(1.0 + r.K_fs - sv, 1e-2), lV, hV))
        fl, uo, va = g.Fluids.values, g.UO.values, g.Vaso.values
        Vs, Ces = np.empty(len(g)), np.empty(len(g))
        for k in range(len(g)):
            Vs[k], Ces[k] = V, Ce
            for _ in range(n_sub):
                V += h * ((fl[k] + r.UO_base - uo[k]) / r.V_ref
                          - max(V - 1.0, 0.0) / r.tau_leak)
                Ce += h * ((va[k] - Ce) / r.tau_ce)
                V = min(max(V, lV), hV); Ce = min(max(Ce, lC), hC)
        Vs_all.append(Vs); Ce_all.append(Ces)
    df = df.copy()
    df["V"], df["Ce"] = np.concatenate(Vs_all), np.concatenate(Ce_all)
    df["m_V"] = df["m_Ce"] = 0.5
    if verbose:
        print(f"[latent] V  [{df.V.min():.2f},{df.V.max():.2f}] median {df.V.median():.2f}"
              f"   Ce nonzero {100*(df.Ce>1e-4).mean():.1f}%")
    return df


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(PROCESSED_ROOT))
    ap.add_argument("--patients", type=int, default=None)
    ap.add_argument("--w_imputed", type=float, default=0.3,
                    help="Weight for GAN-imputed values (M==0) in data loss. "
                         "1.0=treat as observations  0.3=recommended  0.0=real measurements only")
    a = ap.parse_args()

    out = Path(a.out); (out / "patients").mkdir(parents=True, exist_ok=True)
    df = compute_latents(load_and_clean(a.csv, a.patients))

    S, A = list(STATE_NAMES), list(ACTION_NAMES)
    X = df[S].to_numpy(np.float32)
    U = df[A].to_numpy(np.float32)
    Mr = df[[f"m_{s}" for s in S]].to_numpy(np.float32)
    M = np.where(Mr >= 1.0, 1.0, np.where(Mr <= 0.0, a.w_imputed, Mr)).astype(np.float32)

    scales = dict(
        mean=X.mean(0), std=X.std(0) + 1e-6,
        ascl_mean=U.mean(0), ascl_std=U.std(0) + 1e-6,
        state_min=PHYSICAL_MIN.copy(), state_max=PHYSICAL_MAX.copy(),
        state_names=list(S), action_names=list(A), dt=DT_HOURS,
    )
    np.save(out / "scales.npy", scales, allow_pickle=True)
    print(f"[scales] -> {out/'scales.npy'}")

    for p, g in df.groupby("pid", sort=True):
        idx = g.index.values
        np.savez(out / "patients" / f"P{int(p):04d}.npz",
                 t=g.t.to_numpy(np.float32), X=X[idx], U=U[idx], M=M[idx],
                 pid=np.int64(p))
    print(f"[save  ] {df.pid.nunique()} patient files -> {out/'patients'}")

    df[["pid", "t", *S, *A, *[f"m_{s}" for s in S], "AST"]].to_csv(
        out / "cohort_long.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(
        csv=str(Path(a.csv).resolve()), n_patients=int(df.pid.nunique()),
        w_imputed=a.w_imputed, state_names=S, action_names=A,
        dt_hours=DT_HOURS, weight_kg=REFS.weight_kg), indent=2))

    # ---- Quality check ----
    print("\n" + "=" * 66)
    g = df.groupby("pid")
    print(f"{'State':<7}{'real obs/pt':>13}{'mean':>10}{'std':>10}{'min':>9}{'max':>9}")
    for s in STATE_NAMES:
        c = g[f"m_{s}"].apply(lambda z: (z >= 1).sum())
        print(f"{s:<7}{c.mean():>13.2f}{df[s].mean():>10.2f}{df[s].std():>10.2f}"
              f"{df[s].min():>9.2f}{df[s].max():>9.2f}")
    print()
    for act in ACTION_NAMES:
        s = df[act]; nz = s[s > 0]
        print(f"{act:<7} zero%={100*(s==0).mean():5.1f}  nonzero_mean"
              f"={nz.mean() if len(nz) else 0:8.4f}  n_levels={s.nunique():3d}")


if __name__ == "__main__":
    main()
