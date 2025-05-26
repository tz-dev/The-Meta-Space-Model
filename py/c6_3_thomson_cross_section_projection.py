# ========================================================
# File: c6_3_thomson_cross_section_projection.py
# Purpose: Calculate the Thomson scattering cross section using projected constants
# Method:
#   - Use projected fine-structure constant α, reduced Planck constant ħ, electron mass m_e, and speed of light c
#   - Calculate σ_T = (8π/3) * (α ħ / (m_e c))^2
#   - Compare with official Thomson cross section σ_T_official
# Output:
#   - Print projected and official σ_T (m^2)
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Projected input values ===
alpha_proj = 1 / 137.035999084     # fine-structure constant (projected)
hbar_proj = 1.054571817e-34        # reduced Planck constant [J·s] (projected)
m_e_proj = 9.116693e-31             # electron mass [kg] (projected)
c = 2.99758857e8                    # speed of light [m/s] (projected)

# === Calculate Thomson cross section σ_T ===
sigma_T_proj = (8 * np.pi / 3) * (alpha_proj * hbar_proj / (m_e_proj * c))**2

# === Official reference value ===
sigma_T_official = 6.6524587158e-29  # m^2

# === Relative deviation ===
rel_dev = (sigma_T_proj - sigma_T_official) / sigma_T_official * 100

# === Output ===
print("=== Thomson Cross Section Projection ===")
print(f"Projected σ_T = {sigma_T_proj:.5e} m²")
print(f"Official  σ_T = {sigma_T_official:.5e} m²")
print(f"Relative deviation: {rel_dev:.2f}%")
