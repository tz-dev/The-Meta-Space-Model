# ========================================================
# File: c5_6_rydberg_constant_projection.py
# Purpose: Compute Rydberg constant R∞ using entropic projections of fundamental constants
# Method:
#   - Use projected electron mass, fine-structure constant α, speed of light c, and Planck constant h
#   - Calculate R∞ = α² m_e c / (2 h)
#   - Compare with official CODATA Rydberg constant value
# Output:
#   - Print projected and official Rydberg constant (m⁻¹)
#   - Print relative deviation in percent
# ========================================================

import numpy as np

# Projected constants from previous steps
m_e = 9.116693e-31          # kg (projected electron mass)
alpha = 1 / 137.035999084   # fine-structure constant (official, replace if projected α available)
c = 2.99758857e8            # m/s (projected speed of light)
h = 6.62607015e-34          # J·s (Planck constant)

# Calculate Rydberg constant
Rydberg_projected = (alpha**2 * m_e * c) / (2 * h)

# Official CODATA value
Rydberg_official = 10973731.568160  # m^-1

# Calculate relative deviation (%)
relative_deviation = (Rydberg_projected - Rydberg_official) / Rydberg_official * 100

# Output results
print("=== Rydberg Constant from Projected Constants ===")
print(f"Projected R∞: {Rydberg_projected:.6f} m⁻¹")
print(f"Official  R∞: {Rydberg_official:.6f} m⁻¹")
print(f"Relative deviation: {relative_deviation:.5f} %")
