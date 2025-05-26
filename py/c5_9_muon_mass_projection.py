# ========================================================
# File: c5_9_muon_mass_projection.py
# Purpose: Estimate muon mass from geometric mean of electron and tau masses
# Method:
#   - Use projected electron and tau masses from previous sections
#   - Calculate geometric mean: m_mu = sqrt(m_e * m_tau)
#   - Compare with official muon mass
# Output:
#   - Print projected and official muon mass (kg)
#   - Print relative deviation (%)
# ========================================================

import numpy as np

# === Projected masses from previous sections ===
m_e_proj   = 9.116693e-31    # kg (from C.5.1)
m_tau_proj = 3.89142356e-26  # kg (from C.5.11)

# === Geometric mean calculation ===
m_mu_proj = np.sqrt(m_e_proj * m_tau_proj)

# === Official muon mass (CODATA) ===
m_mu_official = 1.883531627e-28  # kg

# === Relative deviation (%) ===
rel_dev = (m_mu_proj - m_mu_official) / m_mu_official * 100

# === Output ===
print("=== Muon Mass from Entropic Scaling ===")
print(f"Projected m_μ: {m_mu_proj:.8e} kg")
print(f"Official  m_μ: {m_mu_official:.8e} kg")
print(f"Relative deviation: {rel_dev:.5f}%")
