# ========================================================
# File: c2_4_yangmills_dynamics.py
# Purpose: Simulate SU(2)-like Yang–Mills dynamics from entropy field A^a_μ
# Parameters:
#   - g = 0.65 (gauge coupling)
#   - κ = ħ / l_meta (field scale)
# Output:
#   - Energy density histogram per color index
# ========================================================

import numpy as np
import matplotlib.pyplot as plt

S_mean = 2.74309
S_sigma = 0.05894
k_B = 1.380649e-23
hbar = 1.054571817e-34
l_meta = 1.616e-35
kappa = hbar / l_meta
g = 0.65

def structure_constant(a, b, c):
    return (a - b) * (b - c) * (c - a) / 2

N = 1000
dS = np.random.normal(loc=0, scale=S_sigma, size=(3, 4, N))
A = kappa * dS

F = np.zeros((3, 4, 4, N))

for a in range(3):
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            dA = dS[a, mu] - dS[a, nu]
            nonlinear = 0
            for b in range(3):
                for c in range(3):
                    fabc = structure_constant(a, b, c)
                    nonlinear += fabc * A[b, mu] * A[c, nu]
            F[a, mu, nu] = dA + g * nonlinear

energy_density = np.zeros((3, N))
for a in range(3):
    F0i2 = sum(F[a, 0, i]**2 for i in range(1, 4))
    Fmunu2 = sum(F[a, mu, nu]**2 for mu in range(4) for nu in range(mu+1, 4))
    energy_density[a] = F0i2 - 0.25 * Fmunu2

for a in range(3):
    print(f"Mean energy density (color {a}): {np.mean(energy_density[a]):.5e} J/m³")

plt.figure(figsize=(8, 6))
for a in range(3):
    plt.hist(energy_density[a], bins=50, alpha=0.6, label=f'Color {a}')
plt.title('Yang–Mills Energy Density Distribution (SU(2)-like)')
plt.xlabel('Energy Density [J/m³]')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('img/c2_4_yangmills_energy_density_histogram.png')
plt.close()
