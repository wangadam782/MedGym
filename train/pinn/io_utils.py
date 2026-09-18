"""Small I/O helpers shared by PINN command-line entry points."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Same physical bounds as ``scripts/train_pinn.py`` / ICU PINN training.
PINN_PHYSICAL_MIN = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32)
PINN_PHYSICAL_MAX = np.array([100.0, 600.0, 30.0, 15.0, 5000.0, 20.0], dtype=np.float32)


def git_rev(repo_root: Path) -> str:
    """Return the current short Git revision, or an empty string outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def enforce_pinn_physical_bounds(scales: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Intersect data-derived ``state_min`` / ``state_max`` with physical bounds in norm space."""
    from data.config import device

    mean = scales["state_mean"].cpu().numpy().astype(np.float32)
    std = scales["state_std"].cpu().numpy().astype(np.float32)

    phys_min_norm = torch.tensor(
        (PINN_PHYSICAL_MIN - mean) / std,
        dtype=torch.float32,
        device=device,
    )
    phys_max_norm = torch.tensor(
        (PINN_PHYSICAL_MAX - mean) / std,
        dtype=torch.float32,
        device=device,
    )

    sc = dict(scales)
    sc["state_min"] = torch.maximum(scales["state_min"], phys_min_norm)
    sc["state_max"] = torch.minimum(scales["state_max"], phys_max_norm)
    return sc


def scale_payload(scales: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert model scale tensors to the stable downstream ``scales.npy`` schema."""
    return {
        "mean": scales["state_mean"].detach().cpu().numpy().astype(np.float32),
        "std": scales["state_std"].detach().cpu().numpy().astype(np.float32),
        "ascl": scales["action_scale"].detach().cpu().numpy().astype(np.float32),
        "state_min": scales["state_min"].detach().cpu().numpy().astype(np.float32),
        "state_max": scales["state_max"].detach().cpu().numpy().astype(np.float32),
    }


def save_scales(scales: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), scale_payload(scales)) # type: ignore


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
