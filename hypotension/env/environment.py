"""Acute-hypotension ICU environment for LAG-TRPO / TRPO RL.

SMDP structure (mirrors medrl-tacos ICUEnvironment):
    state  s_k = (x_k, t_remain_norm, k_remain_norm)   in R^SMDP_STATE_DIM (12)
    option o_k = (u_tanh [, dt_tanh])
        use_dt=False (fixdt): option_dim = action_dim = 3
        use_dt=True  (vardt): option_dim = action_dim + 1 = 4

Hypotension PINN state (10-dim, normalised):
    idx 0  V      effective circulating volume (norm, 1=euvolaemia)
    idx 1  Ce     vasopressor effect-compartment (mcg/kg/min, norm)
    idx 2  MAP    mean arterial pressure (mmHg)  <- reward variable
    idx 3  PP     pulse pressure (mmHg)
    idx 4  UO     urine output (mL/kg/h)         <- safety constraint
    idx 5  Lac    lactate (mmol/L)               <- safety constraint
    idx 6  Cr     creatinine (mg/dL)
    idx 7  Hep    transaminase ALT equiv (IU/L)
    idx 8  PaO2   arterial O2 partial pressure   <- safety constraint
    idx 9  GCS    Glasgow Coma Scale (3-15)

Hypotension PINN action (3-dim, normalised -> physical via scales):
    idx 0  Fluids   mL/kg/h
    idx 1  Vaso     mcg/kg/min
    idx 2  FiO2     fraction

Reward (MAP-centred):
    Flat-top parabola with peak in [65, 80] mmHg, penalising hypotension,
    excessive vasopressor dose, and lactate rise.

Cost (for Lagrangian constraint in LAG-TRPO):
    Combined: hypertension + lactate + oliguria + hypoxaemia.
    Single scalar cost c_k, kept below cost_limit by Lagrangian multiplier lambda.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from hypotension.data.config import (
    device, state_dim, action_dim,
    MAP_TARGET, MAP_CRITICAL, LACTATE_LIMIT, URINE_LIMIT, PAO2_LIMIT,
)

# ── SMDP dims ─────────────────────────────────────────────────────────────────
SMDP_STATE_DIM    = state_dim + 2        # 10 + 2 = 12
OPTION_ACTION_DIM = action_dim + 1       # 3  + 1 = 4 (full vardt)

# ── State indices ─────────────────────────────────────────────────────────────
_I = dict(V=0, Ce=1, MAP=2, PP=3, UO=4, Lac=5, Cr=6, Hep=7, PaO2=8, GCS=9)

# ── Clinical thresholds (physical units) ──────────────────────────────────────
_MAP_TARGET   = MAP_TARGET       # 65 mmHg
_MAP_CRIT     = MAP_CRITICAL     # 60 mmHg  (severe hypotension)
_LAC_THRESH   = LACTATE_LIMIT    # 2.0 mmol/L
_UO_THRESH    = URINE_LIMIT      # 0.5 mL/kg/h
_PAO2_THRESH  = PAO2_LIMIT       # 60 mmHg


class HypoICUEnvironment:
    """Unified acute-hypotension Option-SMDP environment.

    Parameters
    ----------
    pinn_model      : hypotension PINN instance (already on correct device)
    mean_np         : cohort state mean  (state_dim,)
    std_np          : cohort state std   (state_dim,)
    action_min_pinn : action lower bound in normalised space (action_dim,)
    action_max_pinn : action upper bound in normalised space (action_dim,)
    state_lo_norm   : normalised state lower bound (state_dim,)
    state_hi_norm   : normalised state upper bound (state_dim,)
    action_scale_np : physical action scale for denormalisation (action_dim,)
    max_steps       : K — maximum number of SMDP decisions
    dt_min / dt_max : time-step bounds in hours (vardt only)
    total_time_h    : episode horizon in hours (default 48 h)
    use_dt          : True  -> vardt (TaCoS), False -> fixdt
    fixed_dt        : step size when use_dt=False (defaults to dt_min)
    n_substeps      : Euler substeps inside PINN.step (default 8)
    """

    def __init__(
        self,
        pinn_model,
        mean_np:          np.ndarray,
        std_np:           np.ndarray,
        action_min_pinn:  np.ndarray,
        action_max_pinn:  np.ndarray,
        state_lo_norm:    np.ndarray,
        state_hi_norm:    np.ndarray,
        action_scale_np:  np.ndarray,
        max_steps:        int   = 48,
        dt_min:           float = 1.0,
        dt_max:           float = 6.0,
        total_time_h:     float = 48.0,
        use_dt:           bool  = False,
        fixed_dt:         float | None = None,
        n_substeps:       int   = 8,
    ):
        self.pinn         = pinn_model
        self.pinn.eval()
        self.mean         = mean_np.astype(np.float32)
        self.std          = std_np.astype(np.float32)
        self.action_min   = action_min_pinn.astype(np.float32)
        self.action_max   = action_max_pinn.astype(np.float32)
        self.action_scale = action_scale_np.astype(np.float32)
        self.state_lo_np  = state_lo_norm.astype(np.float32)
        self.state_hi_np  = state_hi_norm.astype(np.float32)
        self.state_lo_t   = torch.tensor(state_lo_norm, dtype=torch.float32, device=device)
        self.state_hi_t   = torch.tensor(state_hi_norm, dtype=torch.float32, device=device)

        self.max_steps    = max_steps
        self.dt_min       = dt_min
        self.dt_max       = dt_max
        self.total_time_h = total_time_h
        self.use_dt       = use_dt
        self.fixed_dt     = float(fixed_dt) if fixed_dt is not None else dt_min
        self.n_substeps   = n_substeps

        self.phys_state    = None
        self.current_t     = 0.0
        self.step_count    = 0
        self.prev_map      = None
        self.prev_lactate  = None

        mode = "vardt" if use_dt else f"fixdt({self.fixed_dt}h)"
        print(f"[HypoEnv] {mode}  K={max_steps}  T={total_time_h}h  "
              f"state={state_dim}  action={action_dim}")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def smdp_state_dim(self) -> int:
        return SMDP_STATE_DIM

    @property
    def option_action_dim(self) -> int:
        return action_dim + 1 if self.use_dt else action_dim

    @property
    def n_costs(self) -> int:
        return 1

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, init_state_norm: np.ndarray = None,
              patient=None) -> np.ndarray:
        """Reset episode.

        Accepts either:
          reset(init_state_norm=arr)  – direct normalised state
          reset(patient=patient_dict) – extract t=0 from patient["data"]
        """
        if init_state_norm is None and patient is not None:
            init_state_norm = patient["data"][0, 1:1 + state_dim].cpu().numpy()
        if init_state_norm is None:
            raise ValueError("reset() requires init_state_norm or patient")
        x = np.asarray(init_state_norm, dtype=np.float32).flatten()[:state_dim]
        self.phys_state   = x.copy()
        self.current_t    = 0.0
        self.step_count   = 0
        self.prev_map     = self._get_map_phys(x)
        self.prev_lactate = self._get_lac_phys(x)
        return self._smdp_state(x)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, option_tanh: np.ndarray):
        """Execute one SMDP option.

        option_tanh in [-1,1]^option_action_dim:
            fixdt -> (u0_tanh, u1_tanh, u2_tanh)
            vardt -> (u0_tanh, u1_tanh, u2_tanh, dt_tanh)
        """
        u_tanh = option_tanh[:action_dim]
        u_norm = self.action_min + (u_tanh + 1.0) / 2.0 * (
            self.action_max - self.action_min)
        u_norm = np.clip(u_norm, self.action_min, self.action_max)

        if self.use_dt:
            dt_tanh = float(option_tanh[action_dim])
            dt_raw  = self.dt_min + (dt_tanh + 1.0) / 2.0 * (
                self.dt_max - self.dt_min)
        else:
            dt_raw = self.fixed_dt

        t_remaining = self.total_time_h - self.current_t
        k_remaining = self.max_steps - self.step_count
        dt = float(np.clip(dt_raw, self.dt_min, min(self.dt_max, t_remaining)))
        if k_remaining == 1:
            dt = t_remaining

        x_before  = self.phys_state.copy()
        x_next    = self._pinn_step(self.phys_state, u_norm, dt)
        x_next   += np.random.normal(0.0, 1e-6, size=x_next.shape)

        self.phys_state  = x_next
        self.current_t  += dt
        self.step_count += 1

        reward, reward_info = self._map_reward(x_next, u_norm, dt)
        cost = self._safety_cost(x_next, dt)
        done = self.current_t >= self.total_time_h - 1e-6

        map_phys = reward_info.get("map", self._get_map_phys(x_next))
        sofa_score = round(max(0.0, (_MAP_TARGET - map_phys) / 10.0), 2)

        info = {
            "cost":     round(cost, 4),
            "sofa":     sofa_score,
            "dt":       round(dt, 3),
            "t":        round(self.current_t, 2),
            "t_remain": round(self.total_time_h - self.current_t, 2),
            "k":        self.step_count,
            "k_remain": k_remaining - 1,
            "u_norm":   u_norm.round(4).tolist(),
            "x_before": x_before,
            **reward_info,
        }
        return self._smdp_state(x_next), reward, done, info

    # ── Reward: MAP-centred ──────────────────────────────────────────────────

    def _map_reward(self, x_norm: np.ndarray, u_norm: np.ndarray, dt: float):
        """MAP-centred reward — flat top in [65, 80] mmHg, parabolic outside.

        MAP  45 -> r_map = -2.0  (clamped, severe hypotension)
        MAP  60 -> r_map =  0.0  (zero crossing, below target)
        MAP 72.5-> r_map = +1.0  (flat-top centre)
        MAP  80 -> r_map = +1.0  (flat-top edge)
        MAP  85 -> r_map =  0.0  (zero crossing, hypertension onset)
        MAP 100 -> r_map = -2.0  (clamped, over-treatment)
        """
        s    = x_norm * self.std + self.mean
        map_ = float(s[_I["MAP"]])
        lac  = float(s[_I["Lac"]])
        vaso_frac = float(np.clip(
            (u_norm[1] - self.action_min[1]) / (self.action_max[1] - self.action_min[1] + 1e-8),
            0.0, 1.0))

        _LO, _HI  = 65.0, 80.0
        _HALF_W   = 12.5
        if _LO <= map_ <= _HI:
            r_map = 1.0
        elif map_ < _LO:
            deviation = (map_ - _LO) / _HALF_W
            r_map = float(max(-2.0, 1.0 - deviation ** 2))
        else:
            deviation = (map_ - _HI) / _HALF_W
            r_map = float(max(-2.0, 1.0 - deviation ** 2))

        r_vaso = -0.8 * vaso_frac ** 2
        r_lac = -0.5 * max(0.0, lac - _LAC_THRESH)
        reward = r_map + r_vaso + r_lac

        self.prev_map     = map_
        self.prev_lactate = lac

        return reward, {
            "map":     round(map_, 1),
            "lactate": round(lac, 2),
            "r_map":   round(r_map,  3),
            "r_vaso":  round(r_vaso, 3),
            "r_lac":   round(r_lac,  3),
        }

    # ── Cost: multi-constraint safety ────────────────────────────────────────

    def _safety_cost(self, x_norm: np.ndarray, dt: float) -> float:
        """Combined safety cost for Lagrangian constraint.

        c_k = c_hyper + c_lac + c_uo + c_pao2,  capped at 5.0.
        LAG-TRPO keeps E[c_k] <= cost_limit.
        """
        s    = x_norm * self.std + self.mean
        map_ = float(s[_I["MAP"]])
        lac  = float(s[_I["Lac"]])
        uo   = float(s[_I["UO"]])
        pao2 = float(s[_I["PaO2"]])

        c_hyper = min(max(0.0, (map_ - 85.0) / 10.0), 2.0)

        if lac < _LAC_THRESH:
            c_lac = 0.0
        elif lac < 4.0:
            c_lac = (lac - _LAC_THRESH) / 2.0
        else:
            c_lac = 1.0 + (lac - 4.0) * 0.5
        c_lac = min(c_lac, 2.0)

        c_uo = min(max(0.0, 1.5 * (_UO_THRESH - uo) / _UO_THRESH), 1.0)
        c_pao2 = min(max(0.0, 1.0 * (_PAO2_THRESH - pao2) / _PAO2_THRESH), 1.0)

        return float(min(c_hyper + c_lac + c_uo + c_pao2, 5.0))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pinn_step(self, x_norm: np.ndarray, u_norm: np.ndarray,
                   dt: float) -> np.ndarray:
        x_t = torch.tensor(x_norm, dtype=torch.float32, device=device).unsqueeze(0)
        a_t = torch.tensor(u_norm, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            x_next = self.pinn.step(
                x_t, a_t, dt,
                self.state_lo_t, self.state_hi_t,
                n_substeps=self.n_substeps,
            )
        x_np = x_next.squeeze(0).cpu().numpy()
        return np.clip(x_np, self.state_lo_np, self.state_hi_np)

    def _smdp_state(self, x: np.ndarray) -> np.ndarray:
        t_remain = self.total_time_h - self.current_t
        k_remain = self.max_steps    - self.step_count
        return np.concatenate([
            x,
            [np.float32(t_remain / self.total_time_h),
             np.float32(k_remain / self.max_steps)],
        ]).astype(np.float32)

    def _get_map_phys(self, x_norm: np.ndarray) -> float:
        return float(x_norm[_I["MAP"]] * self.std[_I["MAP"]] + self.mean[_I["MAP"]])

    def _get_lac_phys(self, x_norm: np.ndarray) -> float:
        return float(x_norm[_I["Lac"]] * self.std[_I["Lac"]] + self.mean[_I["Lac"]])
