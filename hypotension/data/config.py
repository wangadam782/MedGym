"""Global configuration: state/action definitions, physiological constants, paths.

Acute-hypotension instance of MedGym-TaCoS, using the PhysioNet Health Gym
synthetic dataset (Synthetic Acute Hypotension v1.0.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# =============================================================================
# Paths
# =============================================================================
HYPO_ROOT = Path(__file__).resolve().parent.parent   # medrl-tacos/hypotension/
ROOT = HYPO_ROOT.parent                               # medrl-tacos/
DATA_ROOT = HYPO_ROOT / "data"
DEFAULT_CSV = DATA_ROOT / "hypotension_healthgym.csv"
PROCESSED_ROOT = DATA_ROOT / "processed"
PINN_IND_ROOT = ROOT / "checkpoints-hypotension" / "pinn" / "ind"
PINN_CLU_ROOT = ROOT / "checkpoints-hypotension" / "pinn" / "clu"
PINN_POP_ROOT = ROOT / "checkpoints-hypotension" / "pinn" / "pop"
ONLINE_ROOT = ROOT / "checkpoints-hypotension" / "online"
EVAL_ROOT = ROOT / "results" / "hypotension"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# State / action
# =============================================================================
STATE_NAMES = (
    "V",        # 0  -           Effective circulating volume index (1.0 = euvolaemia)
    "Ce",       # 1  mcg/kg/min  Vasopressor effect-compartment concentration
    "MAP",      # 2  mmHg        Mean arterial pressure  ← reward variable
    "PP",       # 3  mmHg        Pulse pressure = SBP - DBP
    "UO",       # 4  mL/kg/h    Urine output            ← safety constraint
    "Lac",      # 5  mmol/L     Lactate                 ← safety constraint
    "Cr",       # 6  mg/dL      Creatinine
    "Hep",      # 7  IU/L       Transaminase (ALT equiv)
    "PaO2",     # 8  mmHg       Arterial O2 partial pressure  ← safety constraint
    "GCS",      # 9  3-15       Glasgow Coma Scale
)
ACTION_NAMES = (
    "Fluids",   # mL/kg/h      Fluid bolus rate
    "Vaso",     # mcg/kg/min   Vasopressor infusion rate
    "FiO2",     # fraction     Inspired oxygen fraction
)
UNOBSERVED = ("V", "Ce")            # Integrated from recorded orders

state_dim = len(STATE_NAMES)        # 10
action_dim = len(ACTION_NAMES)      # 3

SIDX = {n: i for i, n in enumerate(STATE_NAMES)}
AIDX = {n: i for i, n in enumerate(ACTION_NAMES)}


# =============================================================================
# Clinical thresholds (physical units). Shared across reward, safety, ODE.
# =============================================================================
MAP_TARGET = 65.0        # mmHg    Hypotension threshold / renal autoregulation knee
MAP_CRITICAL = 60.0      # mmHg    Severe hypotension
MAP_CBF = 55.0           # mmHg    Cerebral blood-flow autoregulation lower limit
LACTATE_LIMIT = 2.0      # mmol/L  Tissue hypoperfusion
URINE_LIMIT = 0.5        # mL/kg/h Oliguria
PAO2_LIMIT = 60.0        # mmHg    Hypoxaemia


# =============================================================================
# Physiological bounds (physical units).
# =============================================================================
PHYSICAL_MIN = np.array(
    [0.20, 0.00, 35.0, 10.0, 0.0, 0.30, 0.20,    5.0,  35.0,  3.0], np.float32)
PHYSICAL_MAX = np.array(
    [2.50, 2.00, 140.0, 110.0, 20.0, 20.0, 15.0, 2000.0, 500.0, 15.0], np.float32)
CLIP = {n: (float(PHYSICAL_MIN[i]), float(PHYSICAL_MAX[i]))
        for i, n in enumerate(STATE_NAMES)}


# =============================================================================
# Physiological reference values.
# Learnable parameters are p = p_ref * exp(theta), theta initialised to zero,
# so training starts at literature values with physically interpretable units.
# =============================================================================
@dataclass(frozen=True)
class PhysioRefs:
    # Volume
    V_ref: float = 70.0          # mL/kg   Total blood volume (V normalisation base)
    tau_leak: float = 12.0       # h       Capillary leak / third-spacing
    K_fs: float = 0.5            # -       Frank-Starling half-saturation constant

    # Vasopressor PK/PD
    tau_ce: float = 0.25         # h       Effect-compartment time constant
    K_vaso: float = 0.15         # mcg/kg/min  Receptor half-saturation concentration
    E_vaso: float = 35.0         # mmHg    Maximum vasopressor effect
    Ce_gut: float = 0.20         # mcg/kg/min  Mesenteric ischaemia onset dose

    # Circulation
    tau_map: float = 0.35        # h       MAP relaxation toward Windkessel target
    MAP_base: float = 64.0       # mmHg    Baseline MAP
    E_preload: float = 30.0      # mmHg    Stroke-volume contribution to MAP
    PP_base: float = 58.0        # mmHg    Baseline pulse pressure
    E_pp: float = 22.0           # mmHg    Stroke-volume contribution to PP
    E_pp_vaso: float = 12.0      # mmHg    Afterload suppression of PP
    gamma_acid: float = 0.15     # /mmol/L Catecholamine resistance coefficient

    # Renal
    tau_uo: float = 0.5          # h       UO response time
    UO_base: float = 1.36        # mL/kg/h Baseline UO (proxy for maintenance fluids)
    UO_max: float = 1.8          # mL/kg/h Autoregulation plateau UO
    tau_auto: float = 8.0        # mmHg    Autoregulation curve steepness
    K_renal_vc: float = 0.45     # mcg/kg/min  Renal vasoconstriction half-dose
    V_congest: float = 1.3       # -       Venous congestion threshold
    K_congest: float = 0.5       # -       Congestion penalty half-constant

    # Metabolic
    P_lac_base: float = 1.33     # mmol/L/h  Baseline glycolysis
    P_lac: float = 3.0           # mmol/L/h  Extra production under full O2 debt
    P_lac_gut: float = 6.0       # mmol/L/h per (mcg/kg/min)  Mesenteric ischaemia
    CL_lac: float = 0.9          # /h      Hepatic lactate clearance

    # Creatinine
    P_cr: float = 0.055          # mg/dL/h   Muscle production rate
    CL_cr: float = 0.055         # /h        GFR-dependent clearance rate

    # Hepatic
    K_hep_rel: float = 60.0      # IU/L/h    Shock-liver enzyme release rate
    t_half_alt: float = 47.0     # h         ALT half-life

    # Respiratory
    tau_o2: float = 0.50         # h         PaO2 equilibration time
    PB_PH2O: float = 713.0       # mmHg      Atmospheric pressure - water vapour
    R_resp: float = 0.8          # -         Respiratory quotient
    PaCO2: float = 40.0          # mmHg      Assumed normal
    shunt_base: float = 0.665    # -         Baseline shunt fraction
    shunt_edema: float = 0.20    # -         Pulmonary-oedema extra shunt
    P50: float = 26.8            # mmHg      Haemoglobin P50
    hill_n: float = 2.7          # -         Hill coefficient

    # Neurological
    tau_gcs_rec: float = 8.0     # h         Consciousness recovery time constant
    K_gcs_inj: float = 0.4       # /mmHg/h   Cerebral ischaemia injury rate

    weight_kg: float = 80.0      # kg        Nominal body weight


REFS = PhysioRefs()

# Data grid
DT_HOURS = 1.0
HORIZON_H = 48.0

# Compatibility stub so medrl-tacos rl/__init__.py can be imported without error.
feature_index = {n: i for i, n in enumerate(STATE_NAMES)}
