# ========================================================
# File: c4_1_rg_flow_unification.py
# Purpose: Compute and plot RG flow of gauge couplings in MSSM, find unification scale
# Method:
#   - Use 1-loop beta coefficients for MSSM gauge groups U(1), SU(2), SU(3)
#   - Calculate inverse couplings α_i^-1(μ) over large energy scale
#   - Identify energy μ where couplings unify approximately
# Output:
#   - Prints unification scale and coupling constant
#   - Saves plot of α_i^-1 vs μ (log scale)
# ========================================================

import numpy as np
import matplotlib.pyplot as plt

mu = np.logspace(2, 17, 500)
log_mu = np.log(mu / 91.2)

b1, b2, b3 = 6.6, 1.0, -3.0
alpha1_0, alpha2_0, alpha3_0 = 1/59.0, 1/29.6, 1/8.5

def alpha_inv(alpha0, b, log_mu):
    return 1/alpha0 - b / (2 * np.pi) * log_mu

alpha1_inv = alpha_inv(alpha1_0, b1, log_mu)
alpha2_inv = alpha_inv(alpha2_0, b2, log_mu)
alpha3_inv = alpha_inv(alpha3_0, b3, log_mu)

diff12 = np.abs(alpha1_inv - alpha2_inv)
diff23 = np.abs(alpha2_inv - alpha3_inv)
diff13 = np.abs(alpha1_inv - alpha3_inv)
total_diff = diff12 + diff23 + diff13

min_idx = np.argmin(total_diff)
mu_unif = mu[min_idx]
alpha_unif = np.mean([alpha1_inv[min_idx], alpha2_inv[min_idx], alpha3_inv[min_idx]])

print("=== Coupling Unification Point ===")
print(f"μ_unif ≈ {mu_unif:.3e} GeV")
print(f"α⁻¹_unif ≈ {alpha_unif:.3f}")
print(f"α_unif ≈ {1/alpha_unif:.5f}")

plt.figure(figsize=(8, 6))
plt.plot(mu, alpha1_inv, label=r'$\alpha_1^{-1}$ (U(1))')
plt.plot(mu, alpha2_inv, label=r'$\alpha_2^{-1}$ (SU(2))')
plt.plot(mu, alpha3_inv, label=r'$\alpha_3^{-1}$ (SU(3))')

plt.axvline(mu_unif, color='k', linestyle='--', label=f'Unification scale\n{mu_unif:.2e} GeV')

plt.xscale('log')
plt.xlabel(r'Energy scale $\mu$ [GeV]')
plt.ylabel(r'Inverse coupling $\alpha_i^{-1}$')
plt.title('RG Flow of MSSM Gauge Couplings and Unification Scale')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('mssm_gauge_coupling_unification.png', dpi=300)
plt.show()
