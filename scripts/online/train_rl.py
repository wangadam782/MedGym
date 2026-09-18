#!/usr/bin/env python3
"""train_rl.py — Train one online-RL policy on one PINN-backed ICUEnvironment.

Single job only; sweeps (algos × K × patients × GPUs) belong in
``scripts/sweep_rl.sh``. YAML ``--config`` supplies defaults (see
``configs/online_rl/default.yaml``: ``paths.save_root`` is usually
``results/online``; PINN dirs under ``results/pinn/{population,individual}``).

Example:
    python scripts/train_rl.py --config configs/online_rl/default.yaml \\
        --scope individual --patient_id 200325 --algo sac --K 20 --dt_mode vardt

``scope=individual`` requires ``init_state_norm.npy`` next to that patient's
``pinn.pt`` (no MIMIC trajectory CSV). ``scope=cluster_pooled`` loads a pooled
cluster PINN from ``<pinn_dir>/cluster_<cluster_id>/pinn.pt`` and trains it like
a population policy.

Checkpoints (after training) under ``<save_root>`` from CLI or config:
    <save_root>/population/<algo>/K<K>/<fixdt|vardt>/policy.pt, actor.pt,
        _meta.json, train_log.csv, ...
    <save_root>/cluster_pooled/cluster_<id>/<algo>/K<K>/<fixdt|vardt>/...
    <save_root>/individual/patient_<pid>/<algo>/K<K>/<fixdt|vardt>/...
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for --config. Install it with `pip install pyyaml`."
    ) from e

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from data.config import state_dim, action_dim
from models import PINN
from utils.pinn_scales import (
    load_individual_init_state_norm,
    load_pinn_folder_scales,
)
from rl import (
    ICUEnvironment,
    SAC,
    PPO,
    TRPO,
    LagrangianPPO,
    LagrangianTRPO,
    SMDP_STATE_DIM,
    OPTION_ACTION_DIM,
)


PHYSICAL_MIN = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32)
PHYSICAL_MAX = np.array([100.0, 600.0, 30.0, 15.0, 5000.0, 20.0], dtype=np.float32)

# CPO is not a supported training target; rl.cpo remains for Lagrangian buffers.
ALGO_CHOICES = [
    "sac",
    "ppo",
    "trpo",
    "lagrangian_ppo",
    "lagrangian_trpo",
]


# ─── Config helpers ──────────────────────────────────────────────────────────

def _load_yaml_config(path: str | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: dict, dotted_key: str, default=None):
    cur: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _default_cluster_train_split_csv() -> str:
    return str(
        _ROOT
        / "reproduce"
        / "extra110_cohort_7_train_only"
        / "patient_split_medrl_algorithms_final_mixed.csv"
    )


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


# ─── PINN / env loading ──────────────────────────────────────────────────────

def _load_population_scales(
    pinn_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    """Load ``scales.npy`` / ``cluster_scales.npy`` under ``pinn_dir`` (see ``utils.pinn_scales``)."""
    return load_pinn_folder_scales(pinn_dir, PHYSICAL_MIN, PHYSICAL_MAX)


def _resolve_individual_pinn(pinn_root: Path, patient_id: int) -> Path:
    candidates = [
        pinn_root / f"patient_{patient_id}" / "pinn.pt",
        pinn_root / str(patient_id) / "pinn.pt",
        pinn_root / str(patient_id) / f"patient_{patient_id}" / "pinn.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"PINN not found for patient {patient_id} under {pinn_root}. "
        "Expected patient_<pid>/pinn.pt or <pid>/pinn.pt."
    )


def _resolve_population_pinn(pinn_dir: Path) -> Path:
    path = pinn_dir / "pinn.pt"
    if not path.exists():
        raise FileNotFoundError(f"Population PINN not found: {path}")
    return path


def _resolve_cluster_pooled_dir(pinn_root: Path, cluster_id: int) -> Path:
    path = pinn_root / f"cluster_{cluster_id}"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Cluster-pooled PINN dir not found for cluster {cluster_id}: {path}"
        )
    return path


def _as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() not in {"0", "false", "f", "no", "n"}


def _read_cluster_train_patient_ids(split_csv: Path, cluster_id: int) -> list[int]:
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Cluster train split CSV not found: {split_csv}. "
            "Pass --cluster_train_split_csv for cluster_pooled RL."
        )
    out: list[int] = []
    with split_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"cluster_id", "patient_id", "split"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{split_csv}: missing required columns: {missing}")
        for row in reader:
            try:
                row_cluster = int(row["cluster_id"])
                patient_id = int(row["patient_id"])
            except (TypeError, ValueError):
                continue
            if row_cluster != int(cluster_id):
                continue
            if str(row.get("split", "")).strip().lower() != "train":
                continue
            if "active_for_run" in row and not _as_bool(row.get("active_for_run")):
                continue
            out.append(patient_id)
    if not out:
        raise RuntimeError(f"No active train patients found for cluster {cluster_id} in {split_csv}")
    return out


def _load_cluster_train_init_states(
    split_csv: Path,
    individual_pinn_root: Path,
    cluster_id: int,
    cluster_mean_np: np.ndarray,
    cluster_std_np: np.ndarray,
    cluster_state_min_np: np.ndarray,
    cluster_state_max_np: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    patient_ids = _read_cluster_train_patient_ids(split_csv, cluster_id)
    init_rows: list[np.ndarray] = []
    for pid in patient_ids:
        init_patient, _, pat_mean, pat_std, _, _, _ = _load_individual_env_inputs(
            individual_pinn_root,
            pid,
        )
        x_phys = init_patient * pat_std[:state_dim] + pat_mean[:state_dim]
        x_cluster = (x_phys - cluster_mean_np[:state_dim]) / cluster_std_np[:state_dim]
        x_cluster = np.clip(
            x_cluster,
            cluster_state_min_np[:state_dim],
            cluster_state_max_np[:state_dim],
        ).astype(np.float32)
        init_rows.append(x_cluster)

    return np.stack(init_rows, axis=0).astype(np.float32), patient_ids


def _load_individual_env_inputs(
    pinn_root: Path,
    patient_id: int,
) -> tuple[np.ndarray, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-patient scales + ``init_state_norm.npy`` only (no trajectory CSV)."""
    pinn_path = _resolve_individual_pinn(pinn_root, patient_id)
    mean_np, std_np, ascl_np, state_min_np, state_max_np, _ = _load_population_scales(
        pinn_path.parent
    )
    init_state_norm = load_individual_init_state_norm(
        pinn_path.parent,
        state_dim=state_dim,
        state_min_np=state_min_np,
        state_max_np=state_max_np,
    )
    if init_state_norm is None:
        raise FileNotFoundError(
            f"Missing init_state_norm.npy next to pinn.pt for patient {patient_id} "
            f"under {pinn_root}. Individual training no longer reads MIMIC trajectory CSV; "
            "run scripts/train_pinn.py or scripts/online/write_pinn_scales.py / "
            "scripts/online/backfill_init_state_norm.py as appropriate."
        )
    init_state_norm = np.asarray(init_state_norm, dtype=np.float32)

    return init_state_norm, pinn_path, mean_np, std_np, ascl_np, state_min_np, state_max_np


def _build_env(
    pinn_model: PINN,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    ascl_np: np.ndarray,
    state_min_np: np.ndarray,
    state_max_np: np.ndarray,
    K: int,
    use_dt: bool,
    total_time_h: float,
    dt_min: float,
    dt_max: float,
    use_lac_penalty: bool,
) -> ICUEnvironment:
    env = ICUEnvironment(
        pinn_model=pinn_model,
        mean_np=mean_np,
        std_np=std_np,
        action_min_norm=np.zeros(action_dim, dtype=np.float32),
        action_max_norm=np.ones(action_dim, dtype=np.float32),
        state_min=state_min_np,
        state_max=state_max_np,
        action_scale_np=ascl_np,
        max_steps=K,
        dt_min=dt_min,
        dt_max=dt_max,
        total_time_h=total_time_h,
        use_dt=use_dt,
        fixed_dt=total_time_h / K,
        use_lac_penalty=use_lac_penalty,
    )

    # Some existing training utilities expect these attributes.
    env.mean_np = mean_np
    env.std_np = std_np
    env.action_scale_np = ascl_np
    return env


# ─── Agent / training ────────────────────────────────────────────────────────

def _build_agent(args: argparse.Namespace, env: ICUEnvironment):
    if args.algo == "sac":
        return SAC(
            state_dim=SMDP_STATE_DIM,
            action_dim=env.option_action_dim,
            gamma=args.gamma,
            tau=args.sac_tau,
            lr=args.sac_lr,
            hidden=args.hidden,
        )

    if args.algo == "ppo":
        return PPO(
            state_dim=SMDP_STATE_DIM,
            action_dim=env.option_action_dim,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            lam=args.lam,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            hidden=args.hidden,
        )

    if args.algo == "trpo":
        return TRPO(
            state_dim=SMDP_STATE_DIM,
            action_dim=env.option_action_dim,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            lam=args.lam,
            delta=args.delta,
            hidden=args.hidden,
        )

    if args.algo == "lagrangian_ppo":
        return LagrangianPPO(
            state_dim=SMDP_STATE_DIM,
            action_dim=OPTION_ACTION_DIM,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            lam=args.lam,
            clip_eps=args.clip_eps,
            hidden=args.hidden,
            cost_limit=args.cost_limit,
            lr_lagrange=args.lr_lagrange,
        )

    if args.algo == "lagrangian_trpo":
        return LagrangianTRPO(
            state_dim=SMDP_STATE_DIM,
            action_dim=OPTION_ACTION_DIM,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            lam=args.lam,
            delta=args.delta,
            hidden=args.hidden,
            cost_limit=args.cost_limit,
            lr_lagrange=args.lr_lagrange,
        )

    raise ValueError(f"Unknown algo: {args.algo}")


def _train_agent(
    args: argparse.Namespace,
    agent,
    env,
    init_state_norm: np.ndarray | None,
    save_dir: Path,
):
    if args.algo == "sac":
        try:
            from train.online_rl.train_sac import train_sac
        except Exception:
            from train import train_sac

        train_sac(
            agent=agent,
            env=env,
            patient=None,
            init_state_norm=init_state_norm,
            total_steps=args.sac_total_steps,
            batch_size=args.sac_batch_size,
            start_steps=args.sac_start_steps,
            save_dir=str(save_dir),
            tag=args.dt_mode,
        )
        return

    if args.algo in ("ppo", "trpo"):
        try:
            from train.online_rl.train_ppo import train_ppo
        except Exception:
            from train import train_ppo

        train_ppo(
            agent=agent,
            env=env,
            patient=None,
            init_state_norm=init_state_norm,
            total_steps=args.on_total_steps,
            rollout_len=args.rollout_len,
            save_dir=str(save_dir),
            tag=args.dt_mode,
            algo_name=args.algo,
        )
        return

    if args.algo in ("lagrangian_ppo", "lagrangian_trpo"):
        try:
            from train.online_rl.train_onpolicy import train_onpolicy_tacos
        except Exception:
            try:
                from train.online_rl.train_onpolicy_tacos import train_onpolicy_tacos
            except Exception:
                from train import train_onpolicy_tacos

        train_onpolicy_tacos(
            agent=agent,
            env=env,
            patients=None,
            init_state_norm=init_state_norm,
            total_steps=args.on_total_steps,
            rollout_len=args.rollout_len,
            save_dir=str(save_dir),
            agent_name=f"{args.algo.upper()}_{args.dt_mode}",
            mean_np=env.mean_np,
            std_np=env.std_np,
            action_scale_np=env.action_scale_np,
        )
        return

    raise ValueError(f"Unknown algo: {args.algo}")


# ─── Saving ──────────────────────────────────────────────────────────────────

def _state_dict_or_none(obj):
    return obj.state_dict() if obj is not None and hasattr(obj, "state_dict") else None


def _extract_extra_state_dicts(agent) -> dict[str, Any]:
    keys = [
        "critic",
        "critic1",
        "critic2",
        "value",
        "value_net",
        "q1",
        "q2",
        "log_alpha",
        "log_lambda",
        "lagrange_multiplier",
    ]
    out = {}
    for key in keys:
        if hasattr(agent, key):
            val = getattr(agent, key)
            sd = _state_dict_or_none(val)
            if sd is not None:
                out[key] = sd
            elif isinstance(val, (float, int)):
                out[key] = val
            elif isinstance(val, torch.Tensor):
                out[key] = val.detach().cpu()
    return out


def _standard_save(
    args: argparse.Namespace,
    agent,
    save_dir: Path,
    pinn_path: Path,
    scales_path: str | Path | None,
    elapsed_sec: float,
    action_dim_used: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    actor_sd = agent.actor.state_dict() if hasattr(agent, "actor") else None
    if actor_sd is None:
        raise RuntimeError("Agent has no .actor; cannot save standard actor.pt.")

    meta = {
        "algo": args.algo,
        "scope": args.scope,
        "policy_scope": args.scope,
        "patient_id": int(args.patient_id) if args.patient_id is not None else None,
        "training_role": args.scope,
        "cluster_id": int(args.cluster_id) if args.cluster_id is not None else None,
        "K": int(args.K),
        "dt_mode": args.dt_mode,
        "save_tag": _save_tag(args),
        "use_dt": bool(args.dt_mode == "vardt"),
        "state_dim": int(SMDP_STATE_DIM),
        "action_dim": int(action_dim_used),
        "hidden": int(args.hidden),
        "total_time_h": float(args.total_time_h),
        "dt_min": float(args.dt_min),
        "dt_max": float(args.dt_max),
        "use_lac_penalty": bool(args.use_lac_penalty),
        "pinn_dir": str(Path(args.pinn_dir).resolve()),
        "pinn_path": str(Path(pinn_path).resolve()),
        "scales_path": str(Path(scales_path).resolve()) if scales_path is not None else None,
        "csv": str(Path(args.csv).resolve())
        if (args.csv is not None and str(args.csv).strip())
        else None,
        "cluster_train_split_csv": str(Path(args.cluster_train_split_csv).resolve())
        if getattr(args, "cluster_train_split_csv", None)
        else None,
        "cluster_init_pinn_dir": str(Path(args.cluster_init_pinn_dir).resolve())
        if getattr(args, "cluster_init_pinn_dir", None)
        else None,
        "cluster_train_patient_ids": getattr(args, "cluster_train_patient_ids", None),
        "cluster_init_state_count": len(getattr(args, "cluster_train_patient_ids", []) or []),
        "save_dir": str(save_dir.resolve()),
        "checkpoint": "policy.pt",
        "actor": "actor.pt",
        "legacy_actor": f"{args.algo}_{args.dt_mode}.pt",
        "config": str(Path(args.config).resolve()) if args.config else None,
        "elapsed_min": round(elapsed_sec / 60.0, 2),
        "git_rev": _git_rev(),
        "argv": sys.argv,
    }

    ckpt = {
        "meta": meta,
        "algo": args.algo,
        "actor": actor_sd,
        **_extract_extra_state_dicts(agent),
    }

    torch.save(ckpt, save_dir / "policy.pt")
    torch.save(actor_sd, save_dir / "actor.pt")

    with open(save_dir / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    train_log = save_dir / "train_log.csv"
    if not train_log.exists():
        with open(train_log, "w") as f:
            f.write("key,value\n")
            f.write(f"elapsed_min,{meta['elapsed_min']}\n")
            f.write(f"algo,{args.algo}\n")
            f.write(f"scope,{args.scope}\n")
            f.write(f"K,{args.K}\n")
            f.write(f"dt_mode,{args.dt_mode}\n")
            f.write(f"save_tag,{meta['save_tag']}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _save_tag(args: argparse.Namespace) -> str:
    return args.dt_mode


def _default_save_dir(args: argparse.Namespace) -> Path:
    root = Path(args.save_root)
    save_tag = _save_tag(args)
    if args.scope == "population":
        return root / "population" / args.algo / f"K{args.K}" / save_tag
    if args.scope == "cluster_pooled":
        if args.cluster_id is None:
            raise ValueError("--cluster_id is required when --scope cluster_pooled.")
        return (
            root
            / "cluster_pooled"
            / f"cluster_{args.cluster_id}"
            / args.algo
            / f"K{args.K}"
            / save_tag
        )
    if args.patient_id is None:
        raise ValueError("--patient_id is required when --scope individual.")
    return (
        root
        / "individual"
        / f"patient_{args.patient_id}"
        / args.algo
        / f"K{args.K}"
        / save_tag
    )


def _parse() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="YAML config file.")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)

    p = argparse.ArgumentParser(
        description="Train one online RL policy on one population or individual PINN.",
        parents=[pre],
    )

    p.add_argument(
        "--scope",
        choices=["population", "individual", "cluster_pooled"],
        default=_cfg_get(cfg, "sweep.scope", "population"),
    )
    p.add_argument("--patient_id", type=int, default=None)
    p.add_argument("--cluster_id", type=int, default=None)
    p.add_argument("--algo", choices=ALGO_CHOICES, required=True)
    p.add_argument("--K", type=int, required=True, help="Max steps per episode.")
    p.add_argument("--dt_mode", choices=["fixdt", "vardt"], required=True)

    p.add_argument("--pinn_dir", default=None)
    p.add_argument(
        "--csv",
        default=_cfg_get(cfg, "paths.csv", None),
        help="Deprecated/unused for individual scope (init_state_norm.npy only). "
        "Ignored for population scope.",
    )
    p.add_argument("--save_root", default=_cfg_get(cfg, "paths.save_root", "results/policies"))
    p.add_argument("--save_dir", default=None)
    p.add_argument(
        "--cluster_train_split_csv",
        default=_cfg_get(cfg, "paths.cluster_train_split_csv", _default_cluster_train_split_csv()),
        help=(
            "CSV with cluster_id,patient_id,split columns. For scope=cluster_pooled, "
            "rows with split=train are used to sample episode initial states."
        ),
    )
    p.add_argument(
        "--cluster_init_pinn_dir",
        default=_cfg_get(cfg, "paths.individual_pinn_dir", "results/pinn/individual"),
        help="Individual PINN root containing patient_<id>/init_state_norm.npy for cluster train initial states.",
    )

    skip_default = bool(_cfg_get(cfg, "sweep.skip_done", False))
    p.add_argument("--skip_done", action="store_true", default=skip_default)
    p.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    p.add_argument("--total_time_h", type=float, default=_cfg_get(cfg, "env.total_time_h", 96.0))
    p.add_argument("--dt_min", type=float, default=_cfg_get(cfg, "env.dt_min", 0.5))
    p.add_argument("--dt_max", type=float, default=_cfg_get(cfg, "env.dt_max", 36.0))

    use_lac_default = bool(_cfg_get(cfg, "env.use_lac_penalty", True))
    p.add_argument("--use_lac_penalty", action="store_true", default=use_lac_default)
    p.add_argument("--no_lac_penalty", dest="use_lac_penalty", action="store_false")

    p.add_argument("--hidden", type=int, default=_cfg_get(cfg, "rl.hidden", 256))
    p.add_argument("--gamma", type=float, default=_cfg_get(cfg, "rl.gamma", 0.997))

    p.add_argument("--sac_total_steps", type=int, default=_cfg_get(cfg, "sac.total_steps", 100_000))
    p.add_argument("--sac_batch_size", type=int, default=_cfg_get(cfg, "sac.batch_size", 512))
    p.add_argument("--sac_start_steps", type=int, default=_cfg_get(cfg, "sac.start_steps", 20_000))
    p.add_argument("--sac_tau", type=float, default=_cfg_get(cfg, "sac.tau", 0.002))
    p.add_argument("--sac_lr", type=float, default=_cfg_get(cfg, "sac.lr", 1e-4))

    p.add_argument("--on_total_steps", type=int, default=_cfg_get(cfg, "onpolicy.total_steps", 200_000))
    p.add_argument("--rollout_len", type=int, default=_cfg_get(cfg, "onpolicy.rollout_len", 2048))
    p.add_argument("--lam", type=float, default=_cfg_get(cfg, "onpolicy.lam", 0.95))
    p.add_argument("--lr_actor", type=float, default=_cfg_get(cfg, "onpolicy.lr_actor", 3e-4))
    p.add_argument("--lr_critic", type=float, default=_cfg_get(cfg, "onpolicy.lr_critic", 1e-3))
    p.add_argument("--clip_eps", type=float, default=_cfg_get(cfg, "onpolicy.clip_eps", 0.2))
    p.add_argument("--entropy_coef", type=float, default=_cfg_get(cfg, "onpolicy.entropy_coef", 0.01))
    p.add_argument("--delta", type=float, default=_cfg_get(cfg, "onpolicy.delta", 0.01))

    p.add_argument("--cost_limit", type=float, default=_cfg_get(cfg, "lagrangian.cost_limit", 0.1))
    p.add_argument("--lr_lagrange", type=float, default=_cfg_get(cfg, "lagrangian.lr_lagrange", 5e-2))

    args = p.parse_args()
    args.config = pre_args.config
    if not str(args.cluster_train_split_csv).strip():
        args.cluster_train_split_csv = _default_cluster_train_split_csv()

    if args.scope == "individual" and args.patient_id is None:
        raise SystemExit("--patient_id is required when --scope individual.")
    if args.scope == "cluster_pooled" and args.cluster_id is None:
        raise SystemExit("--cluster_id is required when --scope cluster_pooled.")


    if args.pinn_dir is None:
        if args.scope == "population":
            args.pinn_dir = _cfg_get(cfg, "paths.population_pinn_dir", "results/pinn/population")
        elif args.scope == "cluster_pooled":
            args.pinn_dir = _cfg_get(
                cfg,
                "paths.cluster_pooled_pinn_dir",
                "results/pinn/cluster_pooled",
            )
        else:
            args.pinn_dir = _cfg_get(cfg, "paths.individual_pinn_dir", "results/pinn/individual")
    


    return args


def main() -> None:
    args = _parse()

    save_dir = Path(args.save_dir) if args.save_dir else _default_save_dir(args)
    if args.skip_done and (save_dir / "policy.pt").exists():
        print(f"[SKIP] policy.pt already exists: {save_dir / 'policy.pt'}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_dt = args.dt_mode == "vardt"
    pinn_dir = Path(args.pinn_dir)

    print("\n" + "=" * 72)
    print("  train_rl.py")
    print(f"  config     : {args.config}")
    print(f"  scope      : {args.scope}")
    print(f"  patient_id : {args.patient_id}")
    print(f"  cluster_id : {args.cluster_id}")
    print(f"  algo       : {args.algo}")
    print(f"  K          : {args.K}")
    print(f"  dt_mode    : {args.dt_mode}")
    print(f"  save_tag   : {_save_tag(args)}")
    print(f"  pinn_dir   : {pinn_dir}")
    print(f"  save_dir   : {save_dir}")

    if args.scope == "population":
        pinn_path = _resolve_population_pinn(pinn_dir)
        mean_np, std_np, ascl_np, state_min_np, state_max_np, scales_path = \
            _load_population_scales(pinn_dir)
    elif args.scope == "cluster_pooled":
        cluster_dir = _resolve_cluster_pooled_dir(pinn_dir, args.cluster_id)
        pinn_path = _resolve_population_pinn(cluster_dir)
        mean_np, std_np, ascl_np, state_min_np, state_max_np, scales_path = \
            _load_population_scales(cluster_dir)
        init_state_norm, cluster_train_patient_ids = _load_cluster_train_init_states(
            split_csv=Path(args.cluster_train_split_csv),
            individual_pinn_root=Path(args.cluster_init_pinn_dir),
            cluster_id=args.cluster_id,
            cluster_mean_np=mean_np,
            cluster_std_np=std_np,
            cluster_state_min_np=state_min_np,
            cluster_state_max_np=state_max_np,
        )
        args.cluster_train_patient_ids = [int(pid) for pid in cluster_train_patient_ids]
    else:
        init_state_norm, pinn_path, mean_np, std_np, ascl_np, state_min_np, state_max_np = \
            _load_individual_env_inputs(pinn_dir, args.patient_id)
        candidate_scales = pinn_path.parent / "scales.npy"
        scales_path = candidate_scales if candidate_scales.exists() else None

    print(f"[PINN] {pinn_path}")
    print(f"[Scales] {scales_path if scales_path is not None else str(pinn_path.parent)}")
    print(f"[Bounds] state_min={np.round(state_min_np, 3)}")
    print(f"[Bounds] state_max={np.round(state_max_np, 3)}")
    if args.scope == "cluster_pooled":
        print(
            f"[Cluster init] {len(args.cluster_train_patient_ids)} train patients "
            f"from {args.cluster_train_split_csv}: {args.cluster_train_patient_ids}"
        )

    pinn_model = PINN(state_dim, action_dim).to(device)
    pinn_model.load_state_dict(torch.load(str(pinn_path), map_location=device))
    pinn_model.eval()

    env = _build_env(
        pinn_model=pinn_model,
        mean_np=mean_np,
        std_np=std_np,
        ascl_np=ascl_np,
        state_min_np=state_min_np,
        state_max_np=state_max_np,
        K=args.K,
        use_dt=use_dt,
        total_time_h=args.total_time_h,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        use_lac_penalty=args.use_lac_penalty,
    )


    agent = _build_agent(args, env)

    t0 = time.time()
    _train_agent(args, agent, env, init_state_norm, save_dir)
    elapsed = time.time() - t0

    action_dim_used = OPTION_ACTION_DIM if args.algo in {
        "lagrangian_ppo", "lagrangian_trpo",
    } else env.option_action_dim

    _standard_save(
        args=args,
        agent=agent,
        save_dir=save_dir,
        pinn_path=pinn_path,
        scales_path=scales_path,
        elapsed_sec=elapsed,
        action_dim_used=action_dim_used,
    )

    print("\n" + "=" * 72)
    print(f"  DONE  saved → {save_dir}")
    print("  artifacts: policy.pt, actor.pt, _meta.json, train_log.csv")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
