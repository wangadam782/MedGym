"""Global feature definitions and constants."""
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_index = {
    "spo2":      0,
    "pao2":      1,
    "bilirubin": 2,
    "gcs":       3,
    "urine":     4,
    "lactate":   5,
    "fio2":      6,
    "vaso":      7,
    "fluids":    8,
}

state_features  = ["SpO2", "PaO2", "Bilirubin", "GCS", "Urine_Step", "Lactate"]
action_features = ["FiO2", "Vaso_Rate", "Fluids_Step"]
selected_features = state_features + action_features

state_dim  = len(state_features)   # 6
action_dim = len(action_features)  # 3

ZERO_IS_NAN = {"SpO2", "FiO2", "PaO2", "Bilirubin", "GCS"}


PINN_POP_ROOT = "results/pinn/population"
PINN_IND_ROOT = "results/pinn/individual"

POLICY_POP_ROOT = "results/online/population"
POLICY_IND_ROOT = "results/online/individual"

POLICY_TRANSFER_EVAL_ROOT = "results/online/transfer_eval"
