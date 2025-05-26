# ========================================================
# File: c5_7_binding_energy_from_alpha.py
# Purpose: Calculate hydrogen binding energy from projected fine-structure constant and electron mass
# Method:
#   - Use projected fine-structure constant α and electron mass m_e
#   - Calculate absolute binding energy: E_bind = 0.5 * α² * m_e * c²
#   - Convert energy from Joules to electron volts (eV)
#   - Compare with official hydrogen binding energy
# Output:
#   - Print projected binding energy (eV)
#   - Print official binding energy (eV)
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Projected Constants ===
alpha = 1 / 137.035999084  # Fine-structure constant
m_e = 9.116693e-31         # kg (from C.5.1)
c = 2.99758857e8           # m/s (from C.5.2)
J_to_eV = 1.602176634e-19  # Conversion factor Joule to eV

# Binding Energy calculation (absolute value)
E_bind_J = 0.5 * alpha**2 * m_e * c**2
E_bind_eV = E_bind_J / J_to_eV

# Official hydrogen binding energy (positive)
E_official = 13.605693122994  # eV

# Relative deviation (%)
rel_dev = (E_bind_eV - E_official) / E_official * 100

# Output
print("=== Binding Energy from Projected Fine Structure ===")
print(f"Projected E_bind: {E_bind_eV:.12f} eV")
print(f"Official  E_bind: {E_official:.12f} eV")
print(f"Relative deviation: {rel_dev:.5f}%")
