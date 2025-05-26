# ========================================================
# File: c5_5_bohr_radius_projection.py
# Purpose: Calculate Bohr radius a₀ using entropic projections of physical constants
# Method:
#   - Use projected electron mass, vacuum permeability, and speed of light
#   - Calculate ε₀ from μ₀ and c
#   - Compute a₀ = (4πε₀ ħ²) / (m_e e²)
#   - Compare with official Bohr radius value
# Output:
#   - Print projected and official Bohr radius in meters
#   - Print relative deviation in percent
# ========================================================

import numpy as np

# Constants from previous projections
hbar = 1.054571817e-34       # J·s (reduced Planck constant)
m_e = 9.116693e-31           # kg (projected electron mass)
mu_0 = 1.2569187994e-6       # H/m (projected vacuum permeability)
c = 2.99758857e8             # m/s (projected speed of light)
e = 1.602176634e-19          # C (elementary charge)

# Calculate vacuum permittivity ε₀ from μ₀ and c
epsilon_0 = 1 / (mu_0 * c**2)

# Calculate Bohr radius a₀
a0_projected = 4 * np.pi * epsilon_0 * hbar**2 / (m_e * e**2)

# Official Bohr radius (CODATA)
a0_official = 5.29177210903e-11  # meters

# Calculate relative deviation (%)
relative_deviation = (a0_projected - a0_official) / a0_official * 100

# Output results
print("=== Bohr Radius from Entropic Projection ===")
print(f"Projected a₀: {a0_projected:.12e} m")
print(f"Official  a₀: {a0_official:.12e} m")
print(f"Relative deviation: {relative_deviation:.5f} %")
