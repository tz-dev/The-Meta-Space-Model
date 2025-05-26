# ========================================================
# File: c6_5_avogadro_number_projection.py
# Purpose: Estimate Avogadro's number from projected atomic mass unit scaling
# Method:
#   - Use projected atomic mass unit m_u_proj
#   - Calculate N_A_proj = 1 gram / m_u_proj
#   - Compare with official Avogadro number N_A_official
# Output:
#   - Print projected and official N_A
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Projected atomic mass unit and mass of 1 gram ===
m_u_proj = 1.66053906660e-27  # atomic mass unit [kg] (projected or assumed)
mass_gram = 1e-3              # 1 gram in kg

# === Calculate Avogadro number projection ===
N_A_proj = mass_gram / m_u_proj

# === Official Avogadro number ===
N_A_official = 6.02214076e23

# === Relative deviation ===
rel_dev = (N_A_proj - N_A_official) / N_A_official * 100

# === Output ===
print("=== Avogadro Number from Projected Atomic Mass Unit ===")
print(f"Projected N_A: {N_A_proj:.6e}")
print(f"Official  N_A: {N_A_official:.6e}")
print(f"Relative deviation: {rel_dev:.5f}%")
