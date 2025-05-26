# ========================================================
# File: c4_2_rg_flow_top_yukawa.py
# Purpose: Solve RG flow of the top Yukawa coupling in MSSM at 1-loop
# Method:
#   - Use beta function for y_t including gauge coupling running effects
#   - Integrate differential equation over energy scale μ = 10² ... 10¹⁷ GeV
# Output:
#   - Plot y_t(μ) vs μ
#   - Print y_t at M_t and at GUT scale ~2e16 GeV
# ========================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

mu = np.logspace(2, 17, 1000)
log_mu = np.log(mu / 173.0)

y_t0 = 0.935

def alpha_inv(mu, alpha0, b):
    return 1/alpha0 - b / (2*np.pi) * np.log(mu / 91.2)

def g(mu, alpha0, b):
    alpha_inv_val = alpha_inv(mu, alpha0, b)
    alpha_inv_clipped = np.clip(alpha_inv_val, 0.1, 1e5)
    return np.sqrt(4*np.pi / alpha_inv_clipped)

def dydlogmu(log_mu, y):
    mu_val = 173.0 * np.exp(log_mu)
    g1_ = g(mu_val, 1/59.0, 6.6)
    g2_ = g(mu_val, 1/29.6, 1.0)
    g3_ = g(mu_val, 1/8.5, -3.0)
    dy = y[0] / (16*np.pi**2) * (4.5*y[0]**2 - ((17/20)*g1_**2 + (9/4)*g2_**2 + 8*g3_**2))
    return [dy]

sol = solve_ivp(dydlogmu, [log_mu[0], log_mu[-1]], [y_t0], t_eval=log_mu, method='RK45')
y_t = sol.y[0]

plt.figure(figsize=(8, 6))
plt.plot(mu, y_t, label=r'$y_t(\mu)$')
plt.xscale('log')
plt.xlabel(r'$\mu$ (GeV)')
plt.ylabel(r'$y_t$')
plt.title('RG Flow of Top Yukawa Coupling')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("img/c4_2_top_yukawa_rg_flow.png")
plt.close()

mu_gut = 2.035e16
yt_gut = float(np.interp(np.log(mu_gut / 173.0), sol.t, y_t))

print("=== Top Yukawa RG Flow ===")
print(f"y_t(M_t = 173 GeV) = {y_t0}")
print(f"y_t(μ = {mu_gut:.2e} GeV) ≈ {yt_gut:.4f}")
