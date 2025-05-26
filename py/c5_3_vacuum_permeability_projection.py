# ========================================================
# File: c5_3_vacuum_permeability_projection.py
# Purpose: Calculate vacuum permeability μ₀ from entropic projection of speed of light and permittivity
# Method:
#   - Use projected speed of light from previous step
#   - Use known electric constant ε₀
#   - Calculate μ₀ = 1 / (ε₀ * c²)
#   - Compare projected μ₀ with official value
# Output:
#   - Print projected and official μ₀ (in H/m)
#   - Print relative deviation in percent
# ========================================================

import numpy as np

# Projected speed of light (from entropic projection)
c_projected = 2.99758857e8  # m/s

# Electric constant (vacuum permittivity)
epsilon_0 = 8.8541878128e-12  # F/m

# Calculate vacuum permeability (μ₀)
mu_0_projected = 1 / (epsilon_0 * c_projected**2)

# Official vacuum permeability μ₀
mu_0_official = 4e-7 * np.pi  # H/m

# Calculate relative deviation (%)
relative_deviation = (mu_0_projected - mu_0_official) / mu_0_official * 100

# Output results
print("=== Vacuum Permeability from Entropic Projection ===")
print(f"Projected μ₀: {mu_0_projected:.10e} H/m")
print(f"Official  μ₀: {mu_0_official:.10e} H/m")
print(f"Relative deviation: {relative_deviation:.5f} %")
