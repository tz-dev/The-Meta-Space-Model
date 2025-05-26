# ========================================================
# File: c2_6_einstein_equations_from_entropy.py
# Purpose: Derive Einstein tensor Gμν from entropy-derived metric gμν
# Method:
#   - Define S(xμ), compute Hessian → gμν
#   - Compute Γ^λ_{μν}, Riemann, Ricci, Scalar curvature R
#   - Compute Gμν = Rμν - ½ R gμν
# Output:
#   - Printed symbolic Einstein tensor Gμν
# ========================================================

import numpy as np
from numpy.linalg import inv
import sympy as sp

x = sp.symbols('t x y z')
coords = [x[0], x[1], x[2], x[3]]

S_expr = sp.exp(x[0]) + sp.sin(x[1])**2 + sp.cos(x[2]) + x[3]**2
Hessian = sp.hessian(S_expr, coords)
g = sp.Matrix(Hessian)
g_inv = g.inv()

Gamma = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
for l in range(4):
    for m in range(4):
        for n in range(4):
            term = 0
            for k in range(4):
                term += g_inv[l, k] * (
                    sp.diff(g[k, m], coords[n]) +
                    sp.diff(g[k, n], coords[m]) -
                    sp.diff(g[m, n], coords[k])
                )
            Gamma[l][m][n] = sp.simplify(0.5 * term)

Riemann = [[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
for rho in range(4):
    for sigma in range(4):
        for mu in range(4):
            for nu in range(4):
                term1 = sp.diff(Gamma[rho][sigma][nu], coords[mu])
                term2 = sp.diff(Gamma[rho][sigma][mu], coords[nu])
                term3 = sum(Gamma[rho][mu][k] * Gamma[k][sigma][nu] for k in range(4))
                term4 = sum(Gamma[rho][nu][k] * Gamma[k][sigma][mu] for k in range(4))
                Riemann[rho][sigma][mu][nu] = sp.simplify(term1 - term2 + term3 - term4)

Ricci = sp.zeros(4)
for mu in range(4):
    for nu in range(4):
        Ricci[mu, nu] = sum(Riemann[l][mu][l][nu] for l in range(4))
Ricci = sp.simplify(Ricci)

Ricci_scalar = sum(g_inv[i, j] * Ricci[i, j] for i in range(4) for j in range(4))
Ricci_scalar = sp.simplify(Ricci_scalar)

Einstein = Ricci - 0.5 * Ricci_scalar * g
Einstein = sp.simplify(Einstein)

print("=== Einstein Tensor from Entropic Metric ===")
sp.pprint(Einstein)
