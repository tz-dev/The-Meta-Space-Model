# ========================================================
# File: c5_11_tau_mass_projection.py
# Purpose: Project tau lepton mass using entropic scaling proportional to electron mass and fine-structure constant
# Method:
#   - Use projected electron mass from C.5.1
#   - Calculate scaling factor as 25 / α times empirical beta factor
#   - Compute tau mass projection as m_e * scale * β_tau
#   - Compare with official tau mass
# Output:
#   - Print projected and official tau mass (kg)
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Input values ===
m_e = 9.116693e-31         # kg (from C.5.1)
alpha = 1 / 137.035999     # fine-structure constant
beta_tau = 1.015           # empirical factor aligned to real tau mass

# === Calculation (25 / alpha scaling) ===
scale = 25 / alpha
m_tau_proj = m_e * scale * beta_tau

# === Official tau mass (CODATA) ===
m_tau_official = 3.16754e-27  # kg

# Relative deviation (%)
rel_dev = (m_tau_proj - m_tau_official) / m_tau_official * 100

# Output
print("=== Tau Mass from Entropic Scaling (25/alpha Model) ===")
print(f"Projected m_tau: {m_tau_proj:.8e} kg")
print(f"Official  m_tau: {m_tau_official:.8e} kg")
print(f"Relative deviation: {rel_dev:.5f}%")
