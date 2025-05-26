# ========================================================
# File: c4_4_rg_stability_param_mapping.py
# Purpose: Analyze RG flow stability and map coupling constants at M_Z scale
# Method:
#   - Compute 1-loop RG running of gauge couplings α_i(μ) over large scale
#   - Calculate derivatives to check flow stability
#   - Map and print coupling values at M_Z and maximum slope of RG flow
# Output:
#   - Print α_i(M_Z), max RG slope
#   - Save plot of α_i(μ) over μ
# ========================================================

import numpy as np
import matplotlib.pyplot as plt

mu = np.logspace(2, 17, 500)
log_mu = np.log(mu / 91.2)

b1, b2, b3 = 6.6, 1.0, -3.0
alpha1_0, alpha2_0, alpha3_0 = 1/59.0, 1/29.6, 1/8.5

def alpha_inv(alpha0, b, log_mu):
    return 1/alpha0 - b / (2 * np.pi) * log_mu

def alpha_val(alpha_inv):
    return 1 / np.clip(alpha_inv, 1e-6, 1e6)

alpha1_inv = alpha_inv(alpha1_0, b1, log_mu)
alpha2_inv = alpha_inv(alpha2_0, b2, log_mu)
alpha3_inv = alpha_inv(alpha3_0, b3, log_mu)

alpha1 = alpha_val(alpha1_inv)
alpha2 = alpha_val(alpha2_inv)
alpha3 = alpha_val(alpha3_inv)

dalpha1 = np.gradient(alpha1, log_mu)
dalpha2 = np.gradient(alpha2, log_mu)
dalpha3 = np.gradient(alpha3, log_mu)

max_slope = max(np.max(np.abs(dalpha1)), np.max(np.abs(dalpha2)), np.max(np.abs(dalpha3)))

print("=== RG Stability and Mapping Check ===")
print(f"α_1(M_Z) ≈ {alpha1[0]:.5f}")
print(f"α_2(M_Z) ≈ {alpha2[0]:.5f}")
print(f"α_3(M_Z) ≈ {alpha3[0]:.5f}")
print(f"Maximum RG flow slope over μ: {max_slope:.5f}")

plt.figure(figsize=(8, 6))
plt.plot(mu, alpha1, label=r'$\alpha_1$ (U(1))')
plt.plot(mu, alpha2, label=r'$\alpha_2$ (SU(2))')
plt.plot(mu, alpha3, label=r'$\alpha_3$ (SU(3))')
plt.xscale('log')
plt.xlabel(r'$\mu$ (GeV)')
plt.ylabel(r'$\alpha_i(\mu)$')
plt.title('RG Flow and Stability of Gauge Couplings')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("img/c4_4_rg_stability_mapping.png")
plt.close()
