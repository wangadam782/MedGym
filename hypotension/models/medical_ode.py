"""Physics prior ODE (MedicalODE) + residual NeuralODE.

Design: separate truly accumulated quantities from algebraic outputs.

  Accumulated (conservation laws or first-order dynamics):
      V    Effective circulating volume  — mass conservation: in - out
      Ce   Vasopressor effect-compartment  — first-order PK
      Lac  Lactate  — mass conservation: anaerobic production - hepatic clearance
      Cr   Creatinine  — mass conservation: muscle production - GFR clearance
      Hep  Transaminase  — enzyme kinetics: injury release - first-order clearance
      GCS  Consciousness  — hypoxic injury accumulation - recovery

  Algebraic outputs (instantaneously determined, expressed as relaxation toward target):
      MAP  = MAP*(V, Ce, Lac)   Windkessel: MAP ~ CO x SVR
      PP   = PP*(V, Ce)         PP proportional to SV / arterial compliance
      UO   = UO*(MAP, V, Ce)    Renal autoregulation curve
      PaO2 = PaO2*(FiO2, V)    Alveolar gas equation + pulmonary-oedema shunt

Three action-cost pathways (all required to prevent "max-dose = optimal"):
    vaso  -> mesenteric ischaemia -> Lac up -> catecholamine resistance -> MAP down
    vaso  -> renal vasoconstriction -> UO down
    fluid -> venous congestion -> UO down and pulmonary shunt up -> PaO2 down

Parameterisation: p = p_ref * exp(theta), theta zero-initialised.
=> Training starts at literature reference values; each parameter has physical
   units and a physiologically interpretable range.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from hypotension.data.config import (
    ACTION_NAMES, CLIP, LACTATE_LIMIT, MAP_CBF, MAP_CRITICAL, MAP_TARGET,
    REFS, STATE_NAMES, action_dim, state_dim,
)


# =============================================================================
# Physiological operators (algebraic, no learnable parameters)
# =============================================================================
def frank_starling(V, K):
    """Normalised stroke volume, SV(1)=1, saturates with V.

    This curve replaces a hand-crafted sigmoid gate: when volume is adequate
    dSV/dV -> 0 and fluid loading naturally becomes ineffective.
    """
    Vp = torch.clamp(V, min=1e-3)
    return (Vp / (Vp + K)) * (1.0 + K)


def sao2_from_pao2(pao2, P50, n):
    p = torch.clamp(pao2, min=1.0).pow(n)
    return p / (p + P50 ** n)


def alveolar_pao2(fio2, r):
    """Alveolar gas equation: PAO2 = FiO2 x (Pb - PH2O) - PaCO2 / R."""
    return fio2 * r.PB_PH2O - r.PaCO2 / r.R_resp


def renal_autoregulation(MAP, knee, tau):
    """Renal autoregulation: GFR maintained above the knee, steeply drops below."""
    return torch.sigmoid((MAP - knee) / tau)


# =============================================================================
class MedicalODE(nn.Module):
    """Clinically-grounded learnable ODE, in PHYSICAL units."""

    PARAMS = [
        "tau_leak", "K_fs",
        "tau_ce", "K_vaso", "E_vaso", "Ce_gut",
        "tau_map", "MAP_base", "E_preload", "PP_base", "E_pp", "E_pp_vaso", "gamma_acid",
        "tau_uo", "UO_base", "UO_max", "tau_auto", "K_renal_vc", "K_congest",
        "P_lac_base", "P_lac", "P_lac_gut", "CL_lac",
        "P_cr", "CL_cr",
        "K_hep_rel", "t_half_alt",
        "tau_o2", "shunt_edema",
        "tau_gcs_rec", "K_gcs_inj",
    ]

    def __init__(self, refs=REFS):
        super().__init__()
        self.r = refs
        self.theta = nn.ParameterDict(
            {n: nn.Parameter(torch.tensor(0.0)) for n in self.PARAMS})

    def p(self, name, cap: float = 3.0):
        """p = p_ref * exp(clamp(theta, +-3)), i.e. within ~20x of reference value."""
        return getattr(self.r, name) * torch.exp(torch.clamp(self.theta[name], -cap, cap))

    def targets(self, x, a):
        """All algebraic target values; exposed for unit tests and visualisation."""
        V, Ce, MAP, PaO2, Lac = x[:, 0], x[:, 1], x[:, 2], x[:, 8], x[:, 5]
        fio2 = a[:, 2]
        r = self.r

        SV = frank_starling(V, self.p("K_fs"))
        kappa = 1.0 / (1.0 + self.p("gamma_acid") * torch.relu(Lac - LACTATE_LIMIT))
        vaso_eff = self.p("E_vaso") * Ce / (Ce + self.p("K_vaso")) * kappa

        MAP_star = self.p("MAP_base") + self.p("E_preload") * (SV - 1.0) + vaso_eff
        PP_star = (self.p("PP_base") + self.p("E_pp") * (SV - 1.0)
                   - self.p("E_pp_vaso") * Ce / (Ce + self.p("K_vaso")))

        congest = 1.0 / (1.0 + torch.relu(V - r.V_congest) / self.p("K_congest"))
        renal_vc = 1.0 / (1.0 + Ce / self.p("K_renal_vc"))
        gfr = renal_autoregulation(MAP, MAP_TARGET, self.p("tau_auto")) * congest * renal_vc
        UO_star = self.p("UO_max") * gfr

        shunt = r.shunt_base + self.p("shunt_edema") * (1.0 - congest)
        PaO2_star = torch.clamp(alveolar_pao2(fio2, r) * (1.0 - shunt), 20.0, 600.0)

        return dict(SV=SV, kappa=kappa, MAP=MAP_star, PP=PP_star,
                    UO=UO_star, gfr=gfr, congest=congest, PaO2=PaO2_star)

    def forward(self, x, a):
        """x, a in PHYSICAL units; returns dx/dt (units/hour)."""
        V, Ce, MAP, PP = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        UO, Lac, Cr, Hep, PaO2, GCS = (x[:, 4], x[:, 5], x[:, 6],
                                       x[:, 7], x[:, 8], x[:, 9])
        u_fluid, u_vaso, _ = a[:, 0], a[:, 1], a[:, 2]
        r, tg = self.r, self.targets(x, a)

        # (1) Volume: mass conservation. UO_base proxies maintenance fluids.
        d_V = ((u_fluid + self.p("UO_base") - UO) / r.V_ref
               - torch.relu(V - 1.0) / self.p("tau_leak"))

        # (2) Vasopressor effect-compartment PK
        d_Ce = (u_vaso - Ce) / self.p("tau_ce")

        # (3)(4) Circulation: relax toward Windkessel targets
        d_MAP = (tg["MAP"] - MAP) / self.p("tau_map")
        d_PP = (tg["PP"] - PP) / self.p("tau_map")

        # (5) Urine output: relax toward renal autoregulation curve
        d_UO = (tg["UO"] - UO) / self.p("tau_uo")

        # (6) Lactate: mass conservation. DO2 proportional to CO x CaO2.
        sao2 = sao2_from_pao2(PaO2, r.P50, r.hill_n)
        ref = sao2_from_pao2(torch.full_like(PaO2, 95.0), r.P50, r.hill_n)
        o2_debt = torch.relu(1.0 - tg["SV"] * sao2 / ref)
        perf_debt = torch.relu(MAP_TARGET - MAP) / MAP_TARGET
        hep_ok = 1.0 / (1.0 + torch.relu(Hep - 40.0) / 400.0)
        d_Lac = (self.p("P_lac_base")
                 + self.p("P_lac") * (o2_debt + perf_debt)
                 + self.p("P_lac_gut") * torch.relu(Ce - self.p("Ce_gut"))
                 - self.p("CL_lac") * hep_ok * Lac)

        # (7) Creatinine: textbook mass conservation
        d_Cr = self.p("P_cr") - self.p("CL_cr") * tg["gfr"] * Cr

        # (8) Transaminase: injury release + first-order clearance
        injury = (torch.relu(MAP_CRITICAL - MAP) / MAP_CRITICAL
                  + 0.5 * torch.relu(Lac - LACTATE_LIMIT) / LACTATE_LIMIT)
        d_Hep = (self.p("K_hep_rel") * injury
                 - np.log(2.0) / self.p("t_half_alt") * torch.relu(Hep))

        # (9) PaO2
        d_PaO2 = (tg["PaO2"] - PaO2) / self.p("tau_o2")

        # (10) Consciousness: cerebral ischaemia injury + recovery
        d_GCS = (-self.p("K_gcs_inj") * torch.relu(MAP_CBF - MAP)
                 - 0.15 * torch.relu(Lac - 4.0)
                 + (15.0 - GCS) / self.p("tau_gcs_rec")
                 * renal_autoregulation(MAP, MAP_CBF, 5.0))

        return torch.stack([d_V, d_Ce, d_MAP, d_PP, d_UO,
                            d_Lac, d_Cr, d_Hep, d_PaO2, d_GCS], 1)

    @torch.no_grad()
    def rollout(self, x0, actions, dt=1.0, max_h=0.1):
        """Pure physics rollout (no neural residual), for smoke tests."""
        lo = torch.tensor([CLIP[n][0] for n in STATE_NAMES], dtype=x0.dtype, device=x0.device)
        hi = torch.tensor([CLIP[n][1] for n in STATE_NAMES], dtype=x0.dtype, device=x0.device)
        n = max(1, int(np.ceil(dt / max_h))); h = dt / n
        xs, x = [x0], x0
        for k in range(actions.shape[1]):
            a = actions[:, k]
            for _ in range(n):
                x = torch.max(torch.min(x + self(x, a) * h, hi), lo)
            xs.append(x)
        return torch.stack(xs, 1)

    def learned_params(self) -> dict:
        return {n: float(self.p(n).detach()) for n in self.PARAMS}


# =============================================================================
class MedicalODENorm(nn.Module):
    """Wrap physical-unit MedicalODE into normalised-space forward(x, a) -> dx/dt."""

    def __init__(self, core: MedicalODE, scales: dict):
        super().__init__()
        self.core = core
        for k, v in [("mean", scales["mean"]), ("std", scales["std"]),
                     ("amean", scales["ascl_mean"]), ("astd", scales["ascl_std"])]:
            self.register_buffer(k, torch.as_tensor(np.asarray(v, np.float32)))

    def forward(self, x, a):
        return self.core(x * self.std + self.mean, a * self.astd + self.amean) / self.std


# =============================================================================
class NeuralODE(nn.Module):
    """Data-driven residual ODE to compensate for unmodelled dynamics."""

    def __init__(self, in_state=state_dim, in_action=action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_state + in_action, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, in_state))

    def forward(self, x, a):
        return torch.clamp(self.net(torch.cat([x, a], 1)), -2.0, 2.0)


class PatientNeuralODE(nn.Module):
    """Lightweight per-patient residual ODE."""

    def __init__(self, in_state=state_dim, in_action=action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_state + in_action, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, in_state))

    def forward(self, x, a):
        return torch.clamp(self.net(torch.cat([x, a], 1)), -1.0, 1.0)
