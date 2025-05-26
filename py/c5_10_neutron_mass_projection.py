# ========================================================
# File: c5_10_neutron_mass_projection.py
# Purpose: Calculate neutron mass from proton mass and experimental mass difference using entropic asymmetry
# Method:
#   - Use official proton mass
#   - Convert neutron-proton mass difference from MeV to Joules, then to kg
#   - Calculate projected neutron mass = proton mass + asymmetry mass difference
#   - Compare with official neutron mass
# Output:
#   - Print projected and official neutron mass (kg)
#   - Print relative deviation (%)
# ========================================================

# Constants
c = 2.99792458e8  # speed of light m/s
c2 = c**2

# Proton mass (CODATA)
m_p = 1.67262192369e-27  # kg

# Experimental neutron-proton mass difference (1.293 MeV → Joules → kg)
delta_E = 1.293e6 * 1.60218e-19  # J
delta_m_asym = delta_E / c2       # kg

# Projected neutron mass
m_n_proj = m_p + delta_m_asym

# Official neutron mass (CODATA)
m_n_official = 1.67492749804e-27  # kg

# Relative deviation (%)
rel_dev = (m_n_proj - m_n_official) / m_n_official * 100

# Output
print("=== Neutron Mass from Projected Asymmetry ===")
print(f"Projected m_n: {m_n_proj:.12e} kg")
print(f"Official  m_n: {m_n_official:.12e} kg")
print(f"Relative deviation: {rel_dev:.5f}%")
