# ========================================================
# File: c3_2_topological_protection.py
# Purpose: Test topological protection via Hessian determinant and entropic divergence statistics
# Method:
#   - Generate stochastic Hessian matrices with signature (1+, 3−)
#   - Calculate determinant and trace to probe stability and divergence
# Output:
#   - Prints statistics on non-zero Hessian determinants and stable entropic divergence counts
# ========================================================

import numpy as np

N_samples = 10000
determinants = []
divergences = []

for _ in range(N_samples):
    ev = [1.0, -1.0, -2.0, -3.0] + np.random.normal(0, 0.2, 4)
    Q, _ = np.linalg.qr(np.random.randn(4, 4))
    H = Q @ np.diag(ev) @ Q.T

    det_H = np.linalg.det(H)
    div_S = np.trace(H)  # Approximate entropic divergence

    determinants.append(det_H)
    divergences.append(div_S)

det_nonzero = np.sum(np.abs(determinants) > 1e-12)
stable_div = np.sum(np.abs(divergences) < 1e1)

print("=== Topological Protection Test ===")
print(f"Total samples: {N_samples}")
print(f"Non-zero Hessian determinants: {det_nonzero} ({100 * det_nonzero / N_samples:.2f}%)")
print(f"Stable entropic divergence (|Tr(H)| < 10): {stable_div} ({100 * stable_div / N_samples:.2f}%)")
