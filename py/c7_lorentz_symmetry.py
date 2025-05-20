import sympy as sp

# 1. Symbole und Metrik definieren
phi = sp.symbols('phi', real=True)
eta = sp.diag(-1, 1, 1, 1)

# 2. Lorentz-Boost-Matrix in x-Richtung
Lambda = sp.Matrix([
    [sp.cosh(phi), -sp.sinh(phi), 0, 0],
    [-sp.sinh(phi), sp.cosh(phi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 3. Überprüfung: Λ^T η Λ - η = 0
invariance_check = sp.simplify(Lambda.T * eta * Lambda - eta)
print(invariance_check)  # ergibt Nullmatrix, damit ist die Invarianz gezeigt
