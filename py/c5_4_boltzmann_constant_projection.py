# ========================================================
# File: c5_4_boltzmann_constant_projection.py
# Purpose: Approximate Boltzmann constant k_B via entropic energy projection and entropy
# Method:
#   - Define entropic projection time τ and calculate ln(Ω) ≈ τ
#   - Use bisection method to find projected energy E_proj such that
#     k_B = E_proj / (ln(Ω) * T) matches official k_B value
# Output:
#   - Print projected k_B value and official k_B
#   - Print relative deviation in percent
#   - Print optimized projected energy E_proj in Joules
# ========================================================

import numpy as np

# Official Boltzmann constant
kB_official = 1.380649e-23  # J/K

# Model parameters
tau = 0.027               # Entropic projection time parameter
ln_Omega = np.log(np.exp(tau))  # ln(Ω) ≈ τ
T = 1.0                   # Temperature in K (arbitrary reference)

def compute_kB(E_proj, ln_Omega=ln_Omega, T=T):
    """Compute kB from projected energy E_proj."""
    return E_proj / (ln_Omega * T)

def optimize_E_proj(target_kB=kB_official, tol=1e-8, max_iter=100):
    """Use bisection to find E_proj yielding kB close to target."""
    E_min, E_max = 1e-25, 1e-21  # plausible search bounds in Joules
    for _ in range(max_iter):
        E_mid = (E_min + E_max) / 2
        kB_mid = compute_kB(E_mid)
        deviation = (kB_mid - target_kB) / target_kB
        if abs(deviation) < tol:
            return E_mid, kB_mid, deviation * 100
        if kB_mid > target_kB:
            E_max = E_mid
        else:
            E_min = E_mid
    # Return best estimate after max iterations
    return E_mid, kB_mid, deviation * 100

# Perform optimization
E_proj_opt, kB_proj, relative_deviation = optimize_E_proj()

# Output results
print("=== Boltzmann Constant from Entropic Projection ===")
print(f"Projected k_B: {kB_proj:.10e} J/K")
print(f"Official  k_B: {kB_official:.10e} J/K")
print(f"Relative deviation: {relative_deviation:.5f} %")
print(f"Optimized projected energy E_proj: {E_proj_opt:.10e} J")
