import numpy as np
import matplotlib.pyplot as plt

# Define spatial grid
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Define example entropy field S(x, y, τ) for fixed τ
# This could represent e.g. an entropy potential well
S = np.exp(-X**2 - Y**2) * np.sin(2 * X)

# Compute second derivatives (Hessian: I_{μν})
S_xx = np.gradient(np.gradient(S, axis=1), axis=1)
S_yy = np.gradient(np.gradient(S, axis=0), axis=0)
S_xy = np.gradient(np.gradient(S, axis=0), axis=1)

# Magnitude of curvature (e.g., trace of Hessian)
I_trace = S_xx + S_yy

# Plot informational curvature (trace as scalar curvature approximation)
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, I_trace, levels=40, cmap='plasma')
plt.colorbar(cp, label=r'Trace $I_{\mu}^{\mu}(x, \tau)$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
plt.savefig('7_5_1.png', dpi=300, bbox_inches='tight')
