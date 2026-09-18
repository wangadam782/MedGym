"""Physics prior ODE (MedicalODE) + residual Neural ODE (NeuralODE).

Their sum provides the ODE constraint: dx/dt ≈ MedicalODE + NeuralODE.
Also defines PatientNeuralODE, a lightweight per-patient residual ODE
trained jointly with the shared PINN/MedicalODE.

"""
import torch
import torch.nn as nn

from data.config import state_dim, action_dim


class MedicalODE(nn.Module):
    """Clinically-inspired learnable ODE with physiological coupling."""

    def __init__(self):
        super().__init__()
        self.k_pao2_fio2   = nn.Parameter(torch.tensor(0.0))
        self.k_pao2_lac    = nn.Parameter(torch.tensor(0.0))
        self.k_spo2_pao2   = nn.Parameter(torch.tensor(0.0))
        self.k_gcs_lac     = nn.Parameter(torch.tensor(0.0))
        self.k_gcs_spo2    = nn.Parameter(torch.tensor(0.0))
        self.k_lac_prod    = nn.Parameter(torch.tensor(0.0))
        self.k_lac_clear   = nn.Parameter(torch.tensor(0.0))
        self.k_urine       = nn.Parameter(torch.tensor(0.0))
        self.k_bili_injury = nn.Parameter(torch.tensor(0.0))
        self.k_bili_clear  = nn.Parameter(torch.tensor(0.0))

    def _k(self, param: nn.Parameter, scale: float = 0.5) -> torch.Tensor:
        return torch.sigmoid(param) * scale

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        spo2    = x[:, 0];  pao2  = x[:, 1];  bili    = x[:, 2]
        gcs     = x[:, 3];  urine = x[:, 4];  lactate = x[:, 5]
        fio2    = a[:, 0];  vaso  = a[:, 1];   fluids  = a[:, 2]

        d_pao2 = torch.clamp(
            self._k(self.k_pao2_fio2) * (fio2 - pao2)
            - self._k(self.k_pao2_lac, 0.1) * lactate
            + self._k(self.k_pao2_lac, 0.05) * fluids,
            -2., 2.)

        d_spo2 = torch.clamp(
            self._k(self.k_spo2_pao2, 0.3) * (pao2 - spo2),
            -1., 1.)

        d_gcs = torch.clamp(
            -self._k(self.k_gcs_lac, 0.1) * torch.relu(lactate)
            - self._k(self.k_gcs_spo2, 0.1) * torch.relu(-spo2),
            -0.5, 0.5)

        d_lactate = torch.clamp(
            self._k(self.k_lac_prod, 0.5) * vaso
            + 0.05 * torch.relu(-spo2)
            - self._k(self.k_lac_clear, 0.5) * torch.relu(lactate)
            - self._k(self.k_lac_clear, 0.1) * fluids,
            -2., 2.)

        d_urine = torch.clamp(
            self._k(self.k_urine) * fluids
            - self._k(self.k_urine, 0.1) * torch.relu(lactate)
            - self._k(self.k_urine, 0.1) * vaso,
            -2., 2.)

        d_bili = torch.clamp(
            self._k(self.k_bili_injury, 0.05) * torch.relu(lactate)
            - self._k(self.k_bili_clear, 0.1) * torch.relu(bili),
            -0.2, 0.2)

        return torch.stack(
            [d_spo2, d_pao2, d_bili, d_gcs, d_urine, d_lactate], dim=1)


class NeuralODE(nn.Module):
    """Data-driven residual ODE to compensate for unmodeled dynamics."""

    def __init__(self, in_state_dim: int = state_dim, in_action_dim: int = action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_state_dim + in_action_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),                          nn.Tanh(),
            nn.Linear(128, in_state_dim),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(torch.cat([x, a], dim=1)), -2., 2.)


class PatientNeuralODE(nn.Module):
    """Lightweight per-patient residual ODE.

    Architecture matches ``train_allpatient_pinn.py`` (64-64 Tanh, output
    clamped to [-1, 1]).  One instance is intended to be created per patient
    and trained jointly with the shared PINN/MedicalODE.
    """

    def __init__(self, in_state_dim: int = state_dim, in_action_dim: int = action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_state_dim + in_action_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),                           nn.Tanh(),
            nn.Linear(64, in_state_dim),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(torch.cat([x, a], dim=1)), -1., 1.)
