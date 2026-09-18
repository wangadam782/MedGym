"""ICU Environment — unified class with use_dt flag (TaCoS / fixed-dt).

State:  s_k = (x_k, t_remain_norm, k_remain_norm)  ∈ R^SMDP_STATE_DIM
Option: o_k = (u_tanh [, δt_tanh])                 ∈ R^option_action_dim
  - use_dt=True  : option_action_dim = action_dim + 1  (TaCoS / variable-dt)
  - use_dt=False : option_action_dim = action_dim       (fixed-dt)
"""
import math
import numpy as np
import torch

from data.config import device, state_dim, action_dim, feature_index
from models.noise_model import ConditionalNoiseModel

SMDP_STATE_DIM    = state_dim + 2   # phys=state_dim, t_remain=1, k_remain=1
OPTION_ACTION_DIM = action_dim + 1  # full option dim (use_dt=True)


class ICUEnvironment:
    """Unified ICU Option-SMDP environment.

    Parameters
    ----------
    use_dt          : True → time-adaptive (TaCoS), False → fixed-dt
    fixed_dt        : step size when use_dt=False (defaults to dt_min)
    use_lac_penalty : kept for backward compat; currently unused
    noise_model     : optional ConditionalNoiseModel applied inside PINN integration
    """

    def __init__(
        self,
        pinn_model,
        mean_np:          np.ndarray,
        std_np:           np.ndarray,
        action_min_norm:  np.ndarray,
        action_max_norm:  np.ndarray,
        state_min:        np.ndarray,
        state_max:        np.ndarray,
        action_scale_np:  np.ndarray,
        max_steps:        int   = 48,
        dt_min:           float = 1.0,
        dt_max:           float = 6.0,
        total_time_h:     float = 96.0,
        safety_mode:      str   = "none",
        lipschitz_g:      float = 1.0,
        gp_surrogate      = None,
        use_dt:           bool  = True,
        fixed_dt:         float = None,
        use_lac_penalty:  bool  = True,   # kept for backward compat
        lactate_threshold: float = 1.0,   # unused threshold (kept for compat)
        noise_model: ConditionalNoiseModel | None = None,
    ):
        self.pinn         = pinn_model
        self.pinn.eval()
        self.mean         = mean_np.astype(np.float32)
        self.std          = std_np.astype(np.float32)
        self.max_steps    = max_steps
        self.dt_min       = dt_min
        self.dt_max       = dt_max
        self.total_time_h = total_time_h
        self.action_min   = action_min_norm.astype(np.float32)
        self.action_max   = action_max_norm.astype(np.float32)
        self.action_scale = action_scale_np.astype(np.float32)
        self.state_min_t  = torch.tensor(state_min, dtype=torch.float32).to(device)
        self.state_max_t  = torch.tensor(state_max, dtype=torch.float32).to(device)
        self.state_min_np = state_min.astype(np.float32)
        self.state_max_np = state_max.astype(np.float32)

        self.safety_mode  = safety_mode
        self.lipschitz_g  = lipschitz_g
        self.gp_surrogate = gp_surrogate

        self.use_dt   = use_dt
        self.fixed_dt = float(fixed_dt) if fixed_dt is not None else dt_min

        self.noise_model = noise_model

        self.phys_state    = None
        self.current_t     = 0.0
        self.step_count    = 0
        self.prev_sofa     = None
        self.prev_ctrl     = None
        self.prev_lactate  = None

        print(f"[Env] use_dt={use_dt}" +
              (f"  fixed_dt={self.fixed_dt}h" if not use_dt else
               f"  dt=[{dt_min},{dt_max}]h"))

    # ── Properties ────────────────────────────────────────────

    @property
    def smdp_state_dim(self) -> int:
        return SMDP_STATE_DIM

    @property
    def option_action_dim(self) -> int:
        return action_dim + 1 if self.use_dt else action_dim

    @property
    def n_costs(self) -> int:
        return 1

    # ── Reset ─────────────────────────────────────────────────

    def reset(self, patient=None, init_state_norm=None) -> np.ndarray:
        if patient is not None:
            x = patient["data"][0, 1:1 + state_dim].cpu().numpy()
        elif init_state_norm is not None:
            x = np.array(init_state_norm, dtype=np.float32)
        else:
            raise ValueError("patient or init_state_norm required")

        self.phys_state        = x.copy()
        self.current_t         = 0.0
        self.step_count        = 0
        self.prev_sofa         = None
        self.prev_ctrl         = None
        self.prev_lactate      = None
        self.interaction_times = [0.0]
        return self._make_smdp_state(x)

    # ── Step ──────────────────────────────────────────────────

    def step(self, option_tanh: np.ndarray):
        # ── Action scaling ────────────────────────────────────
        u_tanh = option_tanh[:action_dim]
        u_norm = self.action_min + (u_tanh + 1.0) / 2.0 * (self.action_max - self.action_min)
        u_norm = np.clip(u_norm, self.action_min, self.action_max)

        # ── dt generation ─────────────────────────────────────
        if self.use_dt:
            dt_tanh = float(option_tanh[action_dim])
            dt_raw  = self.dt_min + (dt_tanh + 1.0) / 2.0 * (self.dt_max - self.dt_min)
            dt_raw  = float(np.clip(dt_raw, self.dt_min, self.dt_max))
        else:
            dt_raw = self.fixed_dt

        # ── Time budget constraint ────────────────────────────
        t_remaining = self.total_time_h - self.current_t
        k_remaining = self.max_steps - self.step_count
        dt = float(np.clip(dt_raw, self.dt_min, min(self.dt_max, t_remaining)))

        # Force final step to consume the remaining time horizon
        if k_remaining == 1:
            dt = t_remaining

        # ── State transition via PINN ─────────────────────────
        x_before = self.phys_state.copy()
        x_next   = self._pinn_integrate(self.phys_state, u_norm, dt)

        # Small noise for realism
        x_next += np.random.normal(0.0, 1e-6, size=x_next.shape)

        self.phys_state  = x_next
        self.current_t  += dt
        self.step_count += 1
        self.interaction_times.append(round(self.current_t, 3))

        # ── Reward ────────────────────────────────────────────
        reward_density, info = self._sofa_reward(x_next, u_norm)

        r_delta = 0.0
        if self.prev_sofa is not None:
            r_delta = 4.0 * (self.prev_sofa - info["sofa_smooth"])
        self.prev_sofa = info["sofa_smooth"]

        reward = (
            reward_density * dt * 0.1
            + r_delta
            - 0.15 * dt
            - 0.02 / (dt + 1e-3)
        )
        reward += 0.3

        # ── Cost (lactate safety) ─────────────────────────────
        cost = self._cost(x_next, dt)

        # ── Info ──────────────────────────────────────────────
        info.update({
            "cost":     round(cost, 4),
            "lactate":  round(self._get_lactate_phys(x_next), 2),
            "dt":       round(dt, 3),
            "t":        round(self.current_t, 2),
            "t_remain": round(self.total_time_h - self.current_t, 2),
            "k":        self.step_count,
            "k_remain": self.max_steps - self.step_count,
            "u_norm":   u_norm.round(3).tolist(),
            "x_before": x_before,
        })

        next_smdp = self._make_smdp_state(x_next)

        # Termination: time horizon only (not step count)
        done = self.current_t >= self.total_time_h - 1e-6

        return next_smdp.copy(), reward, done, info

    # ── Cost (lactate-only + trend penalty) ───────────────────

    def _cost(self, state_norm: np.ndarray, dt: float) -> float:
        s   = state_norm * self.std[:state_dim] + self.mean[:state_dim]
        lac = float(s[feature_index["lactate"]])

        if lac < 1.5:
            base = 0.0
        elif lac < 2.0:
            base = 0.3 * (lac - 1.5) / 0.5
        elif lac < 4.0:
            base = 0.3 + 1.7 * ((lac - 2.0) / 2.0) ** 0.7
        else:
            base = 2.0 + (lac - 4.0)

        base = min(base, 3.0)

        trend = 0.0
        if self.prev_lactate is not None:
            delta = lac - self.prev_lactate
            if delta > 0:
                trend = 0.5 * min(delta / 1.5, 1.0)

        self.prev_lactate = lac
        return float(min(base + trend, 3.5))

    def _get_lactate_phys(self, state_norm: np.ndarray) -> float:
        s = state_norm * self.std[:state_dim] + self.mean[:state_dim]
        return float(s[feature_index["lactate"]])

    # ── Internal ──────────────────────────────────────────────

    def _make_smdp_state(self, x: np.ndarray) -> np.ndarray:
        t_remain = self.total_time_h - self.current_t
        k_remain = self.max_steps    - self.step_count
        return np.concatenate([
            x,
            [np.float32(t_remain / self.total_time_h),
             np.float32(k_remain / self.max_steps)],
        ]).astype(np.float32)

    def _pinn_integrate(
        self,
        x0:         np.ndarray,
        u_norm:     np.ndarray,
        dt:         float,
        n_substeps: int = 4,
    ) -> np.ndarray:
        sub_dt = dt / n_substeps
        x_curr = x0.copy()
        a_t    = torch.tensor(u_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(n_substeps):
                x_t    = torch.tensor(x_curr, dtype=torch.float32).unsqueeze(0).to(device)
                x_next = self.pinn.step(
                    x_t, a_t, sub_dt,
                    self.state_min_t, self.state_max_t,
                )
                if self.noise_model is not None:
                    dt_t   = torch.tensor([sub_dt], dtype=torch.float32).to(device)
                    x_next = x_next + self.noise_model.sample_noise(x_t, dt_t)
                x_curr = x_next.squeeze(0).cpu().numpy()
                x_curr = np.clip(x_curr, self.state_min_np, self.state_max_np)

        return x_curr

    def _check_safety_hard(self, x: np.ndarray, dt: float) -> bool:
        g_val = self._safety_score(x)
        h_bar = (
            self.gp_surrogate.predict_upper(x, dt)
            if self.gp_surrogate is not None
            else 0.5 * dt
        )
        return (g_val + self.lipschitz_g * h_bar) > 0.0

    def _safety_score(self, x: np.ndarray) -> float:
        s    = x * self.std[:state_dim] + self.mean[:state_dim]
        spo2 = float(s[feature_index["spo2"]])
        lac  = float(s[feature_index["lactate"]])
        gcs  = float(s[feature_index["gcs"]])
        return float(max(88.0 - spo2, lac - 8.0, 5.0 - gcs))

    def _sofa_reward(self, state_norm: np.ndarray, action_norm: np.ndarray):
        s = state_norm * self.std[:state_dim] + self.mean[:state_dim]
        a = np.expm1(action_norm * self.action_scale)

        pao2  = float(s[feature_index["pao2"]])
        spo2  = float(s[feature_index["spo2"]])
        bili  = float(s[feature_index["bilirubin"]])
        gcs   = float(s[feature_index["gcs"]])
        urine = float(s[feature_index["urine"]])

        fio2 = np.clip(float(a[0]), 21.0, 100.0) / 100.0
        vaso = float(action_norm[1])

        def sig(x):
            return 1.0 / (1.0 + math.exp(-float(np.clip(x, -50, 50))))

        def smooth_step(x, thr, ws, wl, alpha=0.7):
            return alpha * sig((x - thr) / ws) + (1 - alpha) * sig((x - thr) / wl)

        # SpO2 only (no PaO2 component)
        sr = (
            smooth_step(94 - spo2, 0, 1.0, 3.0) +
            smooth_step(90 - spo2, 0, 1.0, 3.0) +
            smooth_step(85 - spo2, 0, 1.0, 3.0) +
            smooth_step(80 - spo2, 0, 1.0, 3.0)
        ) * 0.7

        sl = (
            smooth_step(bili,  1.2, 0.3, 1.5) +
            smooth_step(bili,  2.0, 0.3, 1.5) +
            smooth_step(bili,  6.0, 0.3, 1.5) +
            smooth_step(bili, 12.0, 0.3, 1.5)
        )

        sn = (
            smooth_step(15 - gcs, 0, 0.5, 1.5) +
            smooth_step(13 - gcs, 0, 0.5, 1.5) +
            smooth_step(10 - gcs, 0, 0.5, 1.5) +
            smooth_step( 6 - gcs, 0, 0.5, 1.5)
        )

        sk = (
            smooth_step(500 - urine, 0, 50.0, 120.0) +
            2 * smooth_step(200 - urine, 0, 50.0, 120.0)
        )

        sc = (
            smooth_step(vaso, 0.0,  0.02, 0.1 ) * 2 +
            smooth_step(vaso, 0.1,  0.03, 0.15) +
            smooth_step(vaso, 0.25, 0.05, 0.2 )
        )

        sofa = sr + sl + sn + sk + sc

        reward = -sofa - 0.2 * vaso
        pf     = pao2 / (fio2 + 1e-6)

        return reward, {
            "sofa":        round(sofa, 2),
            "sofa_smooth": sofa,
            "pf":          round(pf, 1),
            "spo2":        round(spo2, 1),
            "gcs":         round(gcs, 1),
            "components":  {
                "sr": round(sr, 2), "sl": round(sl, 2),
                "sn": round(sn, 2), "sk": round(sk, 2),
                "sc": round(sc, 2),
            },
        }


# Backward-compatible alias
TaCoSEnvironment = ICUEnvironment
