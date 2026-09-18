"""Shared PINN path + scales + init for offline scripts (no CSV when artifacts exist).

Uses ``utils.pinn_scales`` next to ``pinn.pt``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root: scripts/offline/pinn_bundle.py -> parents[2] (tests may only add scripts/offline).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from data.config import state_dim as STATE_DIM
from data.loader import load_patients
from utils.pinn_scales import (
    csv_norm_state_to_pinn_norm_state,
    load_individual_init_state_norm,
    load_pinn_folder_scales,
)

# Same physical bounds as scripts/train_pinn.py / scripts/online/train_rl.py
PHYSICAL_MIN = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32)
PHYSICAL_MAX = np.array([100.0, 600.0, 30.0, 15.0, 5000.0, 20.0], dtype=np.float32)


def resolve_individual_pinn(pinn_root: Path, patient_id: int) -> Path | None:
    """Return path to ``pinn.pt`` if found, else ``None``."""
    candidates = [
        pinn_root / f"patient_{patient_id}" / "pinn.pt",
        pinn_root / str(patient_id) / "pinn.pt",
        pinn_root / str(patient_id) / f"patient_{patient_id}" / "pinn.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_patient_ids_from_pinn_dir(pinn_root: Path) -> list[int]:
    """List ICU ids that have ``pinn.pt`` under ``patient_<id>/`` or numeric ``<id>/``."""
    pids: set[int] = set()
    root = Path(pinn_root)
    if not root.is_dir():
        return []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "pinn.pt").exists():
            continue
        if d.name.startswith("patient_"):
            try:
                pids.add(int(d.name.split("_", 1)[1]))
            except ValueError:
                continue
        elif d.name.isdigit():
            pids.add(int(d.name))
    return sorted(pids)

def load_patient_ids_from_file(path: str | Path) -> list[int]:
    """Load patient IDs from a CSV/TXT patient-list file.

    Supported formats:
      - CSV with an ID header such as pid, patient_id, icu_id, icustay_id, stay_id
      - one ID per line
      - whitespace/comma-separated IDs
      - comment lines starting with '#'

    Returns sorted unique integer IDs. This matches the online workflow's
    patient-list driven runs while keeping offline scripts independent of the
    MIMIC CSV.
    """
    import csv

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Patient ID file not found: {p}")

    lines = p.read_text().splitlines()
    nonempty = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not nonempty:
        return []

    id_columns = ["pid", "patient_id", "icu_id", "icustay_id", "stay_id"]
    header_tokens = [token.strip().lower() for token in nonempty[0].split(",")]

    # CSV-header mode.
    if any(token in id_columns for token in header_tokens):
        ids: list[int] = []
        with p.open(newline="") as f:
            reader = csv.DictReader(
                line for line in f if line.strip() and not line.lstrip().startswith("#")
            )
            if reader.fieldnames is None:
                return []
            field_lookup = {name.strip().lower(): name for name in reader.fieldnames}
            id_field = None
            for candidate in id_columns:
                if candidate in field_lookup:
                    id_field = field_lookup[candidate]
                    break
            if id_field is None:
                raise ValueError(
                    f"{p} must contain one of these ID columns: {', '.join(id_columns)}."
                )
            for row in reader:
                raw = str(row.get(id_field, "")).strip()
                if raw:
                    ids.append(int(float(raw)))
        return sorted(set(ids))

    # Plain text / no-header CSV mode.
    ids: list[int] = []
    for line in nonempty:
        for token in line.replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            ids.append(int(float(token)))
    return sorted(set(ids))

def load_init_and_scales_for_patient(
    pinn_root: Path,
    patient_id: int,
    csv_path: str | None,
    *,
    state_dim: int = STATE_DIM,
) -> tuple[np.ndarray, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load PINN scales from folder and init state (file preferred, else CSV mapping).

    Parameters
    ----------
    csv_path
        If ``None`` or empty string, ``init_state_norm.npy`` must exist under the PINN parent.
    """
    csv_ok = bool(csv_path and str(csv_path).strip())
    pinn_path = resolve_individual_pinn(pinn_root, patient_id)
    if pinn_path is None:
        raise FileNotFoundError(
            f"PINN not found for patient {patient_id} under {pinn_root}."
        )
    parent = pinn_path.parent
    mean_np, std_np, ascl_np, state_min_np, state_max_np, _ = load_pinn_folder_scales(
        parent, PHYSICAL_MIN, PHYSICAL_MAX
    )

    x = load_individual_init_state_norm(
        parent,
        state_dim=state_dim,
        state_min_np=state_min_np,
        state_max_np=state_max_np,
    )
    if x is not None:
        init_norm = np.asarray(x, dtype=np.float32)
    else:
        if not csv_ok:
            raise ValueError(
                f"Patient {patient_id}: no init_state_norm.npy under {parent} and "
                "no MIMIC CSV (--csv) provided for fallback."
            )
        patient, _, scales_csv = load_patients(str(csv_path), patient_id)
        mean_csv = scales_csv["state_mean"].detach().cpu().numpy().astype(np.float32)
        std_csv = scales_csv["state_std"].detach().cpu().numpy().astype(np.float32)
        row0_csv = patient["data"][0, 1 : 1 + state_dim].detach().cpu().numpy().astype(
            np.float32
        )
        init_norm = csv_norm_state_to_pinn_norm_state(
            row0_csv,
            mean_csv,
            std_csv,
            mean_np,
            std_np,
            state_min_np,
            state_max_np,
        )

    return init_norm, pinn_path, mean_np, std_np, ascl_np, state_min_np, state_max_np


def assert_parity_csv_vs_pinn_init(
    csv_path: str,
    pinn_parent: Path,
    patient_id: int,
    *,
    state_dim: int = STATE_DIM,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Raise AssertionError if ``init_state_norm.npy`` disagrees with CSV-mapped init."""
    pinn_parent = Path(pinn_parent)
    mean_np, std_np, _, state_min_np, state_max_np, _ = load_pinn_folder_scales(
        pinn_parent, PHYSICAL_MIN, PHYSICAL_MAX
    )
    x_file = load_individual_init_state_norm(
        pinn_parent,
        state_dim=state_dim,
        state_min_np=state_min_np,
        state_max_np=state_max_np,
    )
    if x_file is None:
        raise AssertionError(f"No init_state_norm.npy under {pinn_parent}; nothing to compare.")

    patient, _, scales_csv = load_patients(csv_path, patient_id)
    mean_csv = scales_csv["state_mean"].detach().cpu().numpy().astype(np.float32)
    std_csv = scales_csv["state_std"].detach().cpu().numpy().astype(np.float32)
    row0_csv = patient["data"][0, 1 : 1 + state_dim].detach().cpu().numpy().astype(np.float32)
    x_csv = csv_norm_state_to_pinn_norm_state(
        row0_csv,
        mean_csv,
        std_csv,
        mean_np,
        std_np,
        state_min_np,
        state_max_np,
    )
    if not np.allclose(x_file, x_csv, rtol=rtol, atol=atol):
        diff = np.max(np.abs(x_file.astype(np.float64) - x_csv.astype(np.float64)))
        raise AssertionError(
            f"init_state_norm vs CSV mapping mismatch (max abs diff={diff:g}). "
            f"file={x_file} csv={x_csv}"
        )
