# ========================================================
# File: c6_1_cosmological_constant_from_hubble.py
# Purpose: Calculate cosmological constant Λ from projected Hubble constant
# Method:
#   - Use projected Hubble constant H0_proj from C.6.2
#   - Calculate Λ = 3 * H0_proj^2 / c^2
#   - Compare with official cosmological constant Λ_official
# Output:
#   - Print projected and official Λ (m^-2)
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Physical constants ===
c = 2.99792458e8  # speed of light [m/s]

# === Projected and official Hubble constants ===
H0_proj = 2.26908425e-18    # 1/s (projected)
H0_official = 2.26854594e-18  # 1/s (official)

# === Calculate cosmological constant Λ ===
Lambda_proj = 3 * H0_proj**2 / c**2
Lambda_official = 1.1056e-52  # m^-2 (official value)

# === Relative deviation calculation ===
rel_dev = (Lambda_proj - Lambda_official) / Lambda_official * 100

# === Output ===
print("=== Cosmological Constant from Projected Hubble Constant ===")
print(f"Λ_proj = {Lambda_proj:.10e} m^-2")
print(f"Λ_official = {Lambda_official:.10e} m^-2")
print(f"Relative deviation: {rel_dev:.5f}%")
