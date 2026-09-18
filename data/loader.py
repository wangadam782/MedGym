"""Data loading and preprocessing for ICU patient time-series."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data.config import (
    ZERO_IS_NAN,
    action_dim,
    action_features,
    device,
    state_dim,
    state_features,
)


@dataclass
class PatientBundle:
    patients: list[dict]
    scales: dict


def _nan_column_stat(
    values: np.ndarray,
    reducer,
    default: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute one statistic per column without emitting all-NaN warnings."""
    stats = []
    for col in values.T:
        observed = col[~np.isnan(col)]
        if observed.size == 0:
            stats.append(default)
        else:
            stats.append(float(reducer(observed)))
    return np.asarray(stats, dtype=dtype)


def _prepare_state_values(df: pd.DataFrame) -> np.ndarray:
    sv = df[state_features].values.astype(np.float32)
    for i, feature in enumerate(state_features):
        if feature in ZERO_IS_NAN:
            sv[:, i] = np.where(sv[:, i] == 0, np.nan, sv[:, i])
    return sv


def _prepare_action_values(df: pd.DataFrame) -> np.ndarray:
    av = df[action_features].values.astype(np.float32)
    av = np.where(np.isnan(av), 0.0, av)
    return np.clip(av, 0.0, None)


def _compute_global_scales(
    df: pd.DataFrame,
    state_bounds: str = "minmax",  # "minmax" (default) or "p5p95"
    margin: float = 0.5,           # half-width margin used by p5p95 only
) -> dict:
    """Compute pooled normalization scales over all rows in df."""
    sv = _prepare_state_values(df)

    state_mean = _nan_column_stat(sv, np.mean, default=0.0)
    state_std = _nan_column_stat(sv, np.std, default=1.0)
    state_std = np.where(state_std == 0, 1.0, state_std).astype(np.float32)

    if state_bounds == "p5p95":
        sv_lo = _nan_column_stat(sv, lambda x: np.percentile(x, 5), default=0.0)
        sv_hi = _nan_column_stat(sv, lambda x: np.percentile(x, 95), default=0.0)
        state_min_norm = (sv_lo - state_mean) / state_std - margin
        state_max_norm = (sv_hi - state_mean) / state_std + margin
    elif state_bounds == "minmax":
        state_lo = _nan_column_stat(sv, np.min, default=0.0)
        state_hi = _nan_column_stat(sv, np.max, default=0.0)
        state_min_norm = (state_lo - state_mean) / state_std
        state_max_norm = (state_hi - state_mean) / state_std
    else:
        raise ValueError(
            f"state_bounds must be 'minmax' or 'p5p95', got {state_bounds!r}."
        )

    av = _prepare_action_values(df)
    action_log = np.log1p(av)
    action_scale = np.max(action_log, axis=0)
    action_scale[action_scale == 0] = 1.0

    return {
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "state_min": state_min_norm.astype(np.float32),
        "state_max": state_max_norm.astype(np.float32),
        "action_scale": action_scale.astype(np.float32),
    }


def _build_single_patient(df: pd.DataFrame, scales_np: dict) -> dict | None:
    df = df.sort_values("hours")
    if len(df) < 2:
        return None

    hours = df["hours"].values.astype(np.float32)
    dt_real = np.diff(hours).astype(np.float32)
    dt_real = np.clip(dt_real, 1e-3, 24.0)
    hours_norm = (hours - hours.min()) / (hours.max() - hours.min() + 1e-6)

    sv = _prepare_state_values(df)
    state_mask = (~np.isnan(sv)).astype(np.float32)
    sv = np.where(np.isnan(sv), scales_np["state_mean"], sv)
    sv = (sv - scales_np["state_mean"]) / scales_np["state_std"]

    av = _prepare_action_values(df)
    av = np.log1p(av) / scales_np["action_scale"]

    x = np.concatenate([hours_norm[:, None], sv, av], axis=1)
    mask = np.concatenate(
        [
            np.ones((len(hours_norm), 1), dtype=np.float32),
            state_mask,
            np.ones((len(hours_norm), action_dim), dtype=np.float32),
        ],
        axis=1,
    )

    return {
        "icu_id": int(df["icu_id"].iloc[0]),
        "data": torch.tensor(x, dtype=torch.float32).to(device),
        "mask": torch.tensor(mask, dtype=torch.float32).to(device),
        "dt": torch.tensor(dt_real, dtype=torch.float32).to(device),
        "state_mask_raw": torch.tensor(state_mask, dtype=torch.float32).to(device),
        "length": int(len(hours_norm)),
    }


def load_all_patients(
    csv_path: str,
    max_patients: int | None = None,
    ids:          list[int] | None = None,  # restrict pool and scale computation to these ICU IDs
    min_rows: int = 2,
    state_bounds: str = "minmax",
    margin: float = 0.5,
) -> PatientBundle:
    """Load multiple patients from a CSV and return a PatientBundle."""
    df = pd.read_csv(csv_path)
    available = df["icu_id"].drop_duplicates().tolist()

    if ids is not None:
        wanted = {int(x) for x in ids}
        keep = [pid for pid in available if int(pid) in wanted]
    else:
        keep = list(available)

    if max_patients is not None:
        keep = keep[:max_patients]

    df = df[df["icu_id"].isin(keep)].copy()
    if df.empty:
        raise ValueError("No rows matched the requested patient selection.")

    scales_np = _compute_global_scales(df, state_bounds=state_bounds, margin=margin)
    scales = {
        k: torch.tensor(v, dtype=torch.float32).to(device)
        for k, v in scales_np.items()
    }

    patients: list[dict] = []
    for pid in keep:
        pdf = df[df["icu_id"] == pid]
        if len(pdf) < min_rows:
            continue
        patient = _build_single_patient(pdf, scales_np)
        if patient is None:
            continue
        patients.append(patient)

    return PatientBundle(patients=patients, scales=scales)


def load_patients(csv_path: str, icu_id: int, df_all: pd.DataFrame | None = None):
    """Load one ICU patient with per-patient normalization.

    Pass df_all to avoid re-reading the CSV in tight loops.
    """
    data = df_all if df_all is not None else pd.read_csv(csv_path)
    df = data[data.icu_id == icu_id].sort_values("hours")

    if len(df) < 2:
        raise ValueError(f"ICU {icu_id}: Not enough data (rows={len(df)})")

    hours = df["hours"].values.astype(np.float32)
    dt_real = np.diff(hours).astype(np.float32)
    dt_real = np.clip(dt_real, 1e-3, 24.0)
    hours_norm = (hours - hours.min()) / (hours.max() - hours.min() + 1e-6)

    sv = _prepare_state_values(df)

    state_mask = (~np.isnan(sv)).astype(float)
    state_mean = _nan_column_stat(sv, np.mean, default=0.0)
    state_std = _nan_column_stat(sv, np.std, default=1.0)
    state_std[state_std == 0] = 1.0

    state_min = _nan_column_stat(sv, np.min, default=0.0)
    state_max = _nan_column_stat(sv, np.max, default=0.0)
    state_min_norm = (state_min - state_mean) / state_std
    state_max_norm = (state_max - state_mean) / state_std

    sv = np.where(np.isnan(sv), state_mean, sv)
    sv = (sv - state_mean) / state_std

    av = _prepare_action_values(df)
    action_log = np.log1p(av)
    action_scale = np.max(action_log, axis=0)
    action_scale[action_scale == 0] = 1.0
    av = action_log / action_scale

    mask = np.concatenate([
        np.ones((len(hours_norm), 1)),
        state_mask,
        np.ones((len(hours_norm), action_dim)),
    ], axis=1)

    x = np.concatenate([hours_norm[:, None], sv, av], axis=1)

    patient = {
        "icu_id": icu_id,
        "data": torch.tensor(x, dtype=torch.float32).to(device),
        "mask": torch.tensor(mask, dtype=torch.float32).to(device),
        "dt": torch.tensor(dt_real, dtype=torch.float32).to(device),
        "state_mask_raw": torch.tensor(state_mask, dtype=torch.float32).to(device),
    }

    scales = {
        "state_mean": torch.tensor(state_mean, dtype=torch.float32).to(device),
        "state_std": torch.tensor(state_std, dtype=torch.float32).to(device),
        "state_min": torch.tensor(state_min_norm, dtype=torch.float32).to(device),
        "state_max": torch.tensor(state_max_norm, dtype=torch.float32).to(device),
        "action_scale": torch.tensor(action_scale, dtype=torch.float32).to(device),
    }

    return patient, 1 + state_dim + action_dim, scales


def find_patient_by_id(patients: list, icu_id: int) -> dict:
    for p in patients:
        if p["icu_id"] == icu_id:
            return p
    raise ValueError(f"ICU_ID {icu_id} not found")
