import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Effective parameters
lambda_prime = 1.0
v_prime = 1.0

# Coordinates in x, y, tau (τ here acts like 'z')
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Define entropy field components
phi_x = np.tanh(X)           # toy model for φ(x)
chi_y_tau = np.exp(-Y**2)    # toy model for χ(y, τ) at fixed τ

# Combine into full entropy field
S = phi_x * chi_y_tau

# Compute entropic effective potential
V_eff = lambda_prime * (S**2 - v_prime**2)**2

# Plot the 3D potential surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, V_eff, cmap='viridis', edgecolor='none')

ax.set_title("Projected Entropic Potential $V_{\\mathrm{eff}}(x, y, \\tau)$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("$V_{\\mathrm{eff}}$")

plt.tight_layout()
plt.savefig('7_4_1.png', dpi=300, bbox_inches='tight')
