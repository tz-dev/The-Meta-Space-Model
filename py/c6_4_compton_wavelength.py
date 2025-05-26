# ========================================================
# File: c6_4_compton_wavelength.py
# Purpose: Calculate Compton wavelength using projected constants with full Planck constant h
# Method:
#   - Use projected ħ, electron mass m_e, speed of light c
#   - Calculate h = 2π ħ
#   - Calculate λ_C = h / (m_e c)
#   - Compare with official Compton wavelength λ_C_official
# Output:
#   - Print projected and official λ_C (m)
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Projected constants ===
hbar = 1.054571817e-34   # reduced Planck constant [J·s]
m_e = 9.116693e-31       # electron mass [kg]
c = 2.99758857e8         # speed of light [m/s]

# === Calculate full Planck constant h ===
h = 2 * np.pi * hbar

# === Calculate Compton wavelength λ_C ===
lambda_C = h / (m_e * c)

# === Official reference value ===
lambda_C_official = 2.42631023867e-12  # m

# === Relative deviation ===
rel_dev = (lambda_C - lambda_C_official) / lambda_C_official * 100

# === Output ===
print("=== Compton Wavelength Projection (using full h) ===")
print(f"Projected λ_C: {lambda_C:.14e} m")
print(f"Official  λ_C: {lambda_C_official:.14e} m")
print(f"Relative deviation: {rel_dev:.5f}%")
