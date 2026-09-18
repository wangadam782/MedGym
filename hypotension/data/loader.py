"""Patient loading, mirroring the original load_patients interface."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hypotension.data.config import PROCESSED_ROOT, action_dim, device, state_dim


def load_scales(processed_root=PROCESSED_ROOT) -> dict:
    p = Path(processed_root) / "scales.npy"
    if not p.exists():
        raise SystemExit(f"Missing {p}\nRun: python -m hypotension.data.preprocess")
    return np.load(str(p), allow_pickle=True).item()


def list_patients(processed_root=PROCESSED_ROOT) -> list[int]:
    d = Path(processed_root) / "patients"
    return sorted(int(f.stem[1:]) for f in d.glob("P*.npz"))


def load_patients(processed_root=PROCESSED_ROOT, patient_id: int | None = None,
                  train_frac: float = 0.7):
    """Return (patient, all_ids, scales), mirroring the sepsis load_patients interface.

    patient dict keys:
        data  (T, 1 + state_dim + action_dim)  column 0 is time(h), then
              state_dim normalised state columns, then action_dim normalised actions
        raw_X (T, state_dim)   physical-unit states (for evaluation)
        raw_U (T, action_dim)  physical-unit actions
        mask  (T, state_dim)   observation mask
        t     (T,)             timestamps (h)
        pid   int
        n_train int            temporal split point: first n_train steps for training
    """
    scales = load_scales(processed_root)
    ids = list_patients(processed_root)
    if patient_id is None:
        return None, ids, scales
    if patient_id not in ids:
        raise SystemExit(f"Patient {patient_id} not found. Available: {ids[:10]} ...({len(ids)} total)")

    d = np.load(Path(processed_root) / "patients" / f"P{patient_id:04d}.npz")
    X, U, M, t = d["X"], d["U"], d["M"], d["t"]
    Xn = (X - scales["mean"]) / scales["std"]
    Un = (U - scales["ascl_mean"]) / scales["ascl_std"]
    data = np.concatenate([t[:, None], Xn, Un], 1).astype(np.float32)

    tt = lambda z: torch.as_tensor(z, dtype=torch.float32, device=device)  # noqa: E731
    T = len(t)
    n_train = max(2, int(round(T * train_frac)))
    return dict(data=tt(data), raw_X=tt(X), raw_U=tt(U), mask=tt(M), t=tt(t),
                pid=int(patient_id), n_train=n_train, T=T), ids, scales


def split_data(patient: dict):
    """Split patient['data'] into (t, x_norm, u_norm)."""
    d = patient["data"]
    return d[:, 0], d[:, 1:1 + state_dim], d[:, 1 + state_dim:1 + state_dim + action_dim]
