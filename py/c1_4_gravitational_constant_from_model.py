# =============================================================================
# File        : c1_4_gravitational_constant_from_model.py
# Purpose     : Estimate gravitational constant G using model-derived ℏ, c, and Planck scale
# Formula     : G ≈ L_eff² · c³ / ħ
# =============================================================================

import numpy as np

# === Input constants ===
hbar = 1.054453619060e-34       # Reconstructed Planck constant [J·s]
c = 2.997589e8                  # Speed of light [m/s]
kB = 1.380649e-23               # Boltzmann constant [J/K]
L_eff = 1.616255e-35            # Effective correlation length (Planck length) [m]

# === Compute G from scaling relation ===
G_proj = (L_eff**2 * c**3) / hbar
G_official = 6.67430e-11        # CODATA value [m³·kg⁻¹·s⁻²]
rel_dev = (G_proj - G_official) / G_official * 100

# === Output ===
print("=== Gravitational Constant from Entropic Scaling ===")
print(f"L_eff:    {L_eff:.5e} m")
print(f"Projected G: {G_proj:.6e} m³·kg⁻¹·s⁻²")
print(f"Official  G: {G_official:.6e} m³·kg⁻¹·s⁻²")
print(f"Relative deviation: {rel_dev:.5f}%")
