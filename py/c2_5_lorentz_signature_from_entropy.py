# ========================================================
# File: c2_5_lorentz_signature_from_entropy.py
# Purpose: Analyze Hessian signature of entropy function → emergent spacetime metric
# Method:
#   - Compute 4×4 Hessians of S(x) at random points
#   - Count positive vs. negative eigenvalues
# Output:
#   - Signature distribution & eigenvalue histograms
# ========================================================

import numpy as np
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt

N_samples = 1000
grid_size = (4, 4)

def entropy_function(x, tau=5.99):
    return np.exp(tau) - 1 + 0.01 * np.sin(2 * np.pi * x[0]) + \
           0.01 * np.cos(2 * np.pi * x[1]) + 0.01 * np.sin(2 * np.pi * x[2]) + \
           0.01 * np.cos(2 * np.pi * x[3])

def compute_hessian(f, x, h=1e-3):
    dim = len(x)
    H = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            x_ijp = np.array(x)
            x_ijm = np.array(x)
            x_ijp[i] += h
            x_ijp[j] += h
            x_ijm[i] -= h
            x_ijm[j] -= h
            H[i, j] = (f(x_ijp) - 2 * f(x) + f(x_ijm)) / (h ** 2)
    return H

signatures = []
eigenvalues_all = []

for _ in range(N_samples):
    x = np.random.uniform(0, 1, 4)
    H = compute_hessian(entropy_function, x)
    ev = eigvalsh(H)
    signature = (np.sum(ev > 0), np.sum(ev < 0))
    signatures.append(signature)
    eigenvalues_all.append(ev)

signature_counts = {}
for sig in signatures:
    signature_counts[sig] = signature_counts.get(sig, 0) + 1

print("=== Lorentz Signature Analysis ===")
for sig, count in sorted(signature_counts.items()):
    print(f"Signature {sig}: {count} occurrences ({100 * count / N_samples:.2f}%)")

eigenvalues_all = np.array(eigenvalues_all)
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.hist(eigenvalues_all[:, i], bins=60, alpha=0.6, label=f'λ_{i+1}')
plt.title("Eigenvalue Distribution of 4D Entropic Hessian")
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("img/c2_5_entropy_hessian_lorentz_signature.png")
plt.close()
