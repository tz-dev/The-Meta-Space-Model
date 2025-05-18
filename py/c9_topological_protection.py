import sympy as sp

# 1. Define coordinate and integer n
theta, n = sp.symbols('theta n', real=True)
# 2. Define gauge potential A_theta = n
A_theta = n
# 3. Compute line integral around S1
integral = sp.integrate(A_theta, (theta, 0, 2*sp.pi))
print(f"∮₀^{2*sp.pi} A = {integral}")  # yields 2*pi*n

# 4. Assert quantization
quantized = sp.simplify(integral - 2*sp.pi*n)
print(f"Quantization check (should be 0): {quantized}")
