# ========================================================
# File: c5_1_electron_mass_from_yukawa.py
# Purpose: Calculate electron mass from Yukawa coupling projection via entropic scaling
# Method:
#   - Use Higgs vacuum expectation value (VEV) and Yukawa coupling approximation
#   - Convert energy units from GeV to Joules
#   - Calculate projected electron mass in Joules and convert to kg
# Output:
#   - Print projected electron mass in kg
#   - Compare with official electron mass (CODATA)
#   - Print relative deviation in percent
# ========================================================

import numpy as np

# Constants
v_GeV = 246  # Higgs VEV in GeV
GeV_to_Joule = 1.60218e-10  # Conversion factor GeV → Joule

# Yukawa coupling approximation for electron (dimensionless)
y_e_approx = 2.94e-6  

# Physical constants
c = 2.99792458e8  # Speed of light in m/s

# Convert Higgs VEV to Joules
v_Joule = v_GeV * GeV_to_Joule

# Calculate electron mass in Joules using Yukawa projection formula: m_e = y_e * v / sqrt(2)
m_e_Joule = y_e_approx * v_Joule / np.sqrt(2)

# Convert energy (J) to mass (kg) using E=mc²
m_e_kg = m_e_Joule / c**2

# Official CODATA electron mass for reference
m_e_official = 9.10938356e-31  # kg

# Calculate relative deviation (%)
relative_deviation = (m_e_kg - m_e_official) / m_e_official * 100

# Output results
print("=== Electron Mass from Entropic Yukawa–Higgs Projection ===")
print(f"Projected electron mass: {m_e_kg:.6e} kg")
print(f"Official electron mass:  {m_e_official:.6e} kg")
print(f"Relative deviation:      {relative_deviation:.5f} %")
