# =============================================================================
# File        : c1_3_rescaling_comparison.py
# Purpose     : Compare reconstructed ℏ from model constants with the curvature-derived value
#               and compute rescaling factor and deviations from official ℏ.
# Sources     : C.5.5 (a₀), C.5.1 (mₑ), C.5.2 (c), α standard, C.1.3 (ℏ from Hessian)
# =============================================================================

import numpy as np

# === Model-derived constants ===
a0_proj = 5.287529430330e-11      # Bohr radius [m]
m_e_proj = 9.116693e-31           # Electron mass [kg]
c_proj = 2.99758857e8             # Speed of light [m/s]
alpha_proj = 1 / 137.035999084    # Fine structure constant

# === Compute ℏ from model ===
hbar_reconstructed = alpha_proj * a0_proj * m_e_proj * c_proj

# === Compare with curvature-derived value ===
hbar_from_curvature = 2.52263935e-26     # Derived from C.1.3 analysis
hbar_official = 1.054571817e-34          # CODATA value

# === Rescaling and relative errors ===
scaling_factor = hbar_reconstructed / hbar_from_curvature
rescaled_hbar = hbar_from_curvature * scaling_factor

rel_dev_rescaled = (rescaled_hbar - hbar_official) / hbar_official * 100
rel_dev_reconstructed = (hbar_reconstructed - hbar_official) / hbar_official * 100

# === Output summary ===
print(f"a0_proj ..............: {a0_proj}")
print(f"m_e_proj .............: {m_e_proj}")
print(f"c_proj ...............: {c_proj}")
print(f"hbar_reconstructed ...: {hbar_reconstructed}")
print(f"hbar_from_curvature ..: {hbar_from_curvature}")
print(f"hbar_official ........: {hbar_official}")
print(f"scaling_factor .......: {scaling_factor}")
print(f"rescaled_hbar ........: {rescaled_hbar}")
print(f"rel_dev_rescaled .....: {rel_dev_rescaled:.4f}%")
print(f"rel_dev_reconstructed : {rel_dev_reconstructed:.4f}%")
