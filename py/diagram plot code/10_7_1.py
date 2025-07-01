import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Parameters for double-well potential
phi = np.linspace(-2, 2, 100)  # Field parameter
V = phi**4 - 2*phi**2 + 1  # V(phi) = phi^4 - 2*phi^2 + 1

# Plot potential
ax.plot(phi, V, color='blue', label='Potential $V(\\phi)$')

# Mark minima
minima = [-1, 1]
V_minima = [0, 0]
ax.scatter(minima, V_minima, color='red', s=100, label='Stable States')

# Set labels and title
ax.set_xlabel('Field Parameter ($\\phi$)')
ax.set_ylabel('Potential ($V(\\phi)$)')

# Set axis limits
ax.set_xlim([-2, 2])
ax.set_ylim([0, 1.5])

# Add legend
ax.legend(loc='upper center')

# Add grid
ax.grid(True, ls='--')

# Add annotation for projection-induced symmetry breaking
ax.text(0.25, 0.75, 'Projection-induced Symmetry Breaking (CP3, EP11)', transform=ax.transAxes, fontsize=10, color='red')

plt.tight_layout()
plt.show()