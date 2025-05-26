# =============================================================================
# C.1.3 – ℏ Reconstruction from Model Constants
# File: c1_3_hbar_reconstruction_from_model.py
# This script reconstructs Planck’s constant ℏ from model-based estimates of:
# - Bohr radius a₀
# - electron mass mₑ
# - speed of light c
# - fine structure constant α
# ℏ is computed via: ℏ ≈ α × a₀ × mₑ × c
# =============================================================================

import numpy as np

# --- Projected model constants ---
a0_proj = 5.287529430330e-11     # Bohr radius [m] from C.5.5
m_e_proj = 9.116693e-31          # Electron mass [kg] from C.5.1
c_proj = 2.99758857e8            # Speed of light [m/s] from C.5.2
alpha_proj = 1 / 137.035999084   # Fine structure constant (standard)

# --- ℏ reconstruction ---
hbar_proj = alpha_proj * a0_proj * m_e_proj * c_proj
hbar_official = 1.054571817e-34  # J·s
rel_dev = (hbar_proj - hbar_official) / hbar_official * 100

# --- Output ---
print("=== ℏ Reconstruction from Model-Based Constants ===")
print(f"a₀_proj  = {a0_proj:.12e} m")
print(f"mₑ_proj  = {m_e_proj:.12e} kg")
print(f"c_proj   = {c_proj:.6e} m/s")
print(f"α_proj   = {alpha_proj:.12e}")
print(f"ℏ_proj   = {hbar_proj:.12e} J·s")
print(f"ℏ_offic. = {hbar_official:.12e} J·s")
print(f"Relative deviation: {rel_dev:.5f}%")
