# ========================================================
# File: c2_2_entropy_vectorfield.py
# Purpose: Project the entropy gradient ∂μS onto an abelian vector field Aμ
# Parameters:
#   - S_mean = 2.74309 (average entropy)
#   - S_sigma = 0.05894 (entropy standard deviation)
#   - κ = ħ / l_meta (field strength scaling constant)
# Output:
#   - Histograms and norm plot of Aμ components
# ========================================================

import numpy as np
import matplotlib.pyplot as plt

S_mean = 2.74309
S_sigma = 0.05894
k_B = 1.380649e-23
hbar = 1.054571817e-34
l_meta = 1.616e-35
tau_meta = 5.391e-44
c = 2.99792458e8

N = 10000
np.random.seed(42)

dS_dtau = np.random.normal(loc=S_mean, scale=S_sigma, size=N)
dS_dx = np.random.normal(loc=0, scale=S_sigma, size=N)
dS_dy = np.random.normal(loc=0, scale=S_sigma, size=N)
dS_dz = np.random.normal(loc=0, scale=S_sigma, size=N)
grad_S = np.stack([dS_dtau, dS_dx, dS_dy, dS_dz], axis=1)

kappa = hbar / l_meta
A_mu = kappa * grad_S

A_mean = np.mean(A_mu, axis=0)
A_max = np.max(np.abs(A_mu), axis=0)

print("A_mu mean components:", A_mean)
print("A_mu max components:", A_max)

plt.figure(figsize=(10, 6))
labels = [r'$A_0$', r'$A_1$', r'$A_2$', r'$A_3$']
for i in range(4):
    plt.hist(A_mu[:, i], bins=60, alpha=0.6, label=labels[i])
plt.title('Histogram of Projected Abelian Vector Field Components $A_\\mu$')
plt.xlabel('Field strength (J/m)')
plt.ylabel('Density')
plt.legend()
plt.savefig('img/c2_2_entropy_vectorfield_histogram.png')
plt.close()

A_norm = np.linalg.norm(A_mu, axis=1)
plt.figure(figsize=(8, 6))
plt.plot(np.sort(A_norm))
plt.title('Sorted Norm of Vector Field $|A_\\mu|$')
plt.xlabel('Sample Index')
plt.ylabel(r'$|A_\mu|$ (J/m)')
plt.savefig('img/c2_2_entropy_vectorfield_norm_sorted.png')
plt.close()
