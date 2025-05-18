import sympy as sp

# 1. Define coordinates and coefficients
x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3', real=True)
a, b, c, d = sp.symbols('a b c d', real=True)

# 2. Define a simple quadratic entropy function S(x)
S = a*x0**2 + b*x1**2 + c*x2**2 + d*x3**2

# 3. Compute the informational curvature tensor I_{μν} = ∂_μ ∂_ν S
coords = [x0, x1, x2, x3]
I = sp.hessian(S, coords)
print("I_{μν} =")
sp.pprint(I)

# 4. Compute divergence ∇^μ I_{μν} using η^{μσ} ∂_σ I_{μν}
eta = sp.diag(-1, 1, 1, 1)
divergence = [sum(eta[i, i] * sp.diff(I[i, j], coords[i]) for i in range(4)) for j in range(4)]
print("\n∇^μ I_{μν} =")
sp.pprint(divergence)
