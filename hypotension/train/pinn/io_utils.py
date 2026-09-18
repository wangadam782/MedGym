"""I/O helpers: scales, physical bounds, metadata."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from hypotension.data.config import PHYSICAL_MAX, PHYSICAL_MIN

PINN_PHYSICAL_MIN = PHYSICAL_MIN.copy()
PINN_PHYSICAL_MAX = PHYSICAL_MAX.copy()


def enforce_pinn_physical_bounds(scales: dict) -> dict:
    """Write physiological hard bounds into scales for clipping during rollout."""
    s = dict(scales)
    s["state_min"] = np.maximum(
        np.asarray(s.get("state_min", PINN_PHYSICAL_MIN), np.float32), PINN_PHYSICAL_MIN)
    s["state_max"] = np.minimum(
        np.asarray(s.get("state_max", PINN_PHYSICAL_MAX), np.float32), PINN_PHYSICAL_MAX)
    return s


def norm_bounds(scales: dict):
    """Physiological bounds -> normalised space."""
    lo = (scales["state_min"] - scales["mean"]) / scales["std"]
    hi = (scales["state_max"] - scales["mean"]) / scales["std"]
    return lo.astype(np.float32), hi.astype(np.float32)


def save_scales(scales: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), scales, allow_pickle=True)


def save_json(obj: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def git_rev(repo_root) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(repo_root),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
