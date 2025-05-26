# ========================================================
# File: c4_3_yukawa_matrix_eigenflow.py
# Purpose: Compute RG flow of Yukawa matrix eigenvalues for up-type quarks
# Method:
#   - Initialize diagonal Yukawa matrix with realistic quark Yukawas at low scale
#   - Use simplified 1-loop RGE for diagonal elements
#   - Integrate and track eigenvalues over energy scale μ = 10² ... 10¹⁷ GeV
# Output:
#   - Plot eigenvalues λ_i(μ)
#   - Print eigenvalues at GUT scale
# ========================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import eigvalsh

mu_vals = np.logspace(2, 17, 300)
log_mu = np.log(mu_vals / 173.0)

Y0 = np.diag([2e-6, 1e-3, 0.935])  # simplified initial Yukawa matrix

def dyukawa_evolution(log_mu, Y_flat):
    Y = Y_flat.reshape(3, 3)
    y_diag = np.diag(Y)
    A = 0.5
    B = 4.5 / (16*np.pi**2)
    dY = Y * (A - B * y_diag**2)
    return dY.flatten()

sol = solve_ivp(dyukawa_evolution, [log_mu[0], log_mu[-1]], Y0.flatten(),
                t_eval=log_mu, method='RK45')
Y_tensors = sol.y.T.reshape(-1, 3, 3)

eigenvalues = np.array([np.sort(eigvalsh(Y)) for Y in Y_tensors])

plt.figure(figsize=(8, 6))
plt.plot(mu_vals, eigenvalues[:, 0], label=r'$\lambda_1$ (light)')
plt.plot(mu_vals, eigenvalues[:, 1], label=r'$\lambda_2$ (mid)')
plt.plot(mu_vals, eigenvalues[:, 2], label=r'$\lambda_3$ (heavy)')
plt.xscale('log')
plt.xlabel(r'$\mu$ (GeV)')
plt.ylabel(r'Eigenvalue $\lambda_i$')
plt.title('RG Flow of Yukawa Matrix Eigenvalues')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("img/c4_3_yukawa_eigenvalue_flow.png")
plt.close()

mu_gut = 2.035e16
log_mu_gut = np.log(mu_gut / 173.0)
ev_gut = [
    np.interp(log_mu_gut, sol.t, eigenvalues[:, 0]),
    np.interp(log_mu_gut, sol.t, eigenvalues[:, 1]),
    np.interp(log_mu_gut, sol.t, eigenvalues[:, 2]),
]

print("=== Yukawa Matrix RG Eigenvalue Flow ===")
print(f"At μ = {mu_gut:.2e} GeV:")
print(f"λ1 ≈ {ev_gut[0]:.4e}, λ2 ≈ {ev_gut[1]:.4e}, λ3 ≈ {ev_gut[2]:.4e}")
