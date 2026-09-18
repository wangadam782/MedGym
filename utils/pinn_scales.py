"""Load ``scales.npy`` / ``cluster_scales.npy`` next to ``pinn.pt`` (population or individual).

Shared by online and offline scripts. Individual PINN directories may also contain
``init_state_norm.npy`` (shape ``(state_dim,)``), physiological state in the **same
normalized space** as ``scales.npy`` (i.e. after ``(s_phys - mean) / std``, clipped to
``[state_min, state_max]``). Training should write this file once per patient; RL/eval
prefers it over CSV row0 when present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

INIT_STATE_NORM_FILENAME = "init_state_norm.npy"


def load_pinn_folder_scales(
    pinn_parent: Path,
    physical_min: np.ndarray,
    physical_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    """Return ``mean, std, ascl, state_min, state_max`` in PINN normalized space.

    ``pinn_parent`` is the directory containing ``pinn.pt`` and ``scales.npy``.
    """
    candidates = [pinn_parent / "scales.npy", pinn_parent / "cluster_scales.npy"]
    scales_path = next((p for p in candidates if p.exists()), None)
    if scales_path is None:
        raise FileNotFoundError(
            f"No scales file under {pinn_parent}. Expected scales.npy or cluster_scales.npy."
        )

    sc = np.load(str(scales_path), allow_pickle=True).item()
    mean_np = sc["mean"].astype(np.float32)
    std_np = sc["std"].astype(np.float32)
    ascl_np = sc["ascl"].astype(np.float32)

    pmin = np.asarray(physical_min, dtype=np.float32).reshape(-1)
    pmax = np.asarray(physical_max, dtype=np.float32).reshape(-1)
    phys_min_norm = (pmin - mean_np) / std_np
    phys_max_norm = (pmax - mean_np) / std_np

    if "state_min" in sc and "state_max" in sc:
        state_min_np = np.maximum(phys_min_norm, sc["state_min"].astype(np.float32))
        state_max_np = np.minimum(phys_max_norm, sc["state_max"].astype(np.float32))
    else:
        state_min_np = phys_min_norm.astype(np.float32)
        state_max_np = phys_max_norm.astype(np.float32)

    return mean_np, std_np, ascl_np, state_min_np, state_max_np, scales_path


def save_init_state_norm(path: Path, x_norm: np.ndarray) -> None:
    """Save a single-row normalized state vector next to ``pinn.pt`` / ``scales.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(x_norm, dtype=np.float32).reshape(-1)
    np.save(str(path), v, allow_pickle=False)


def load_individual_init_state_norm(
    pinn_parent: Path,
    *,
    state_dim: int,
    state_min_np: np.ndarray,
    state_max_np: np.ndarray,
    filename: str = INIT_STATE_NORM_FILENAME,
) -> np.ndarray | None:
    """Load ``init_state_norm.npy`` if present; clip to env bounds. Otherwise ``None``."""
    p = Path(pinn_parent) / filename
    if not p.exists():
        return None
    x = np.load(str(p), allow_pickle=False).astype(np.float32).reshape(-1)
    if int(x.shape[0]) != int(state_dim):
        raise ValueError(
            f"{p}: expected length {state_dim}, got {x.shape[0]}"
        )
    smin = np.asarray(state_min_np, dtype=np.float32).reshape(-1)
    smax = np.asarray(state_max_np, dtype=np.float32).reshape(-1)
    return np.clip(x, smin, smax).astype(np.float32)


def csv_norm_state_to_pinn_norm_state(
    state_norm_csv: np.ndarray,
    mean_csv: np.ndarray,
    std_csv: np.ndarray,
    mean_npy: np.ndarray,
    std_npy: np.ndarray,
    smin_npy: np.ndarray,
    smax_npy: np.ndarray,
) -> np.ndarray:
    """Map first-row state from ``load_patients`` coords into ``scales.npy`` coords."""
    x_csv = np.asarray(state_norm_csv, dtype=np.float32).reshape(-1)
    m_csv = np.asarray(mean_csv, dtype=np.float32).reshape(-1)
    s_csv = np.asarray(std_csv, dtype=np.float32).reshape(-1)
    s_phys = x_csv * s_csv + m_csv
    x_npy = (s_phys - mean_npy) / std_npy
    return np.clip(x_npy, smin_npy, smax_npy).astype(np.float32)
