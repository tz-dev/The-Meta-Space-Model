# ========================================================
# File: c5_8_proton_mass_from_strong_coupling.py
# Purpose: Estimate proton mass from strong coupling constant α_s at M_Z scale using 1-loop QCD RG running
# Method:
#   - Input α_s at M_Z and number of flavors n_f=3
#   - Calculate QCD scale parameter Λ_QCD from 1-loop RG formula
#   - Approximate proton mass as multiple of Λ_QCD (C_p ≈ 4)
#   - Convert proton mass from GeV to kg
#   - Compare projected proton mass with official CODATA value
# Output:
#   - Print Λ_QCD in GeV
#   - Print projected and official proton mass in kg and GeV
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Input parameters: strong coupling at M_Z scale ===
alpha3_MZ = 0.11704       # α_s at M_Z (from C.4.4)
mu_MZ = 91.2              # M_Z scale in GeV

# === RG parameters for QCD with n_f=3 flavors ===
n_f = 3
b3 = 11.0 - 2.0/3.0 * n_f  # beta-function coefficient = 11 - 2 = 9

# --- Calculate Λ_QCD from 1-loop RG solution ---
# α_s(μ) = 2π / [b3 * ln(μ / Λ_QCD)]  ⇒  Λ_QCD = μ * exp(-2π / (b3 * α_s(μ)))
Lambda_QCD = mu_MZ * np.exp(-2 * np.pi / (b3 * alpha3_MZ))  # in GeV

# --- Proton factor ---
# Approximate proton mass as m_p ≈ C_p * Λ_QCD, with C_p ≈ 4 (phenomenological)
C_p = 4.0

# Proton mass in GeV
m_p_GeV = C_p * Lambda_QCD

# Convert proton mass to kg: 1 GeV/c² ≈ 1.78266192e-27 kg
gev_to_kg = 1.78266192e-27
m_p_proj = m_p_GeV * gev_to_kg

# Official proton mass
m_p_official = 1.67262192369e-27  # kg

# Relative deviation (%)
rel_dev = (m_p_proj - m_p_official) / m_p_official * 100

# Output
print("=== Proton Mass from Strong Coupling Projection (n_f=3) ===")
print(f"Λ_QCD       ≈ {Lambda_QCD:.3f} GeV")
print(f"m_p (proj)  ≈ {m_p_proj:.5e} kg  ({m_p_GeV:.3f} GeV)")
print(f"m_p (official)= {m_p_official:.5e} kg")
print(f"Rel. dev.   = {rel_dev:.2f}%")
