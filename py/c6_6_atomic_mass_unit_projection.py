# ========================================================
# File: c6_6_atomic_mass_unit_projection.py
# Purpose: Calculate atomic mass unit (1u) from projected proton, neutron, and electron masses
# Method:
#   - Use projected m_p, m_n, m_e
#   - Compute C-12 atom mass as sum of 6 protons, 6 neutrons, and 6 electrons
#   - Divide by 12 to get atomic mass unit
#   - Compare with official atomic mass unit
# Output:
#   - Print projected and official 1u (kg)
#   - Print relative deviation (%)
# ========================================================

# === Projected masses ===
m_e_proj = 9.116693e-31      # electron mass [kg]
m_p_proj = 1.66956e-27       # proton mass [kg]
m_n_proj = 1.67492691e-27    # neutron mass [kg]

# === Calculate C-12 atom mass ===
mass_C12 = 6 * (m_p_proj + m_n_proj + m_e_proj)

# === Calculate atomic mass unit (1u) ===
m_u_proj = mass_C12 / 12

# === Official atomic mass unit ===
m_u_official = 1.66053906660e-27  # kg

# === Relative deviation ===
rel_dev = (m_u_proj - m_u_official) / m_u_official * 100

# === Output ===
print("=== Atomic Mass Unit from Projected Masses ===")
print(f"Projected 1u: {m_u_proj:.12e} kg")
print(f"Official  1u: {m_u_official:.12e} kg")
print(f"Relative deviation: {rel_dev:.5f}%")
